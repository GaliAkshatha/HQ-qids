"""
tests/support/local_suzume_target.py

A real, in-process Flask app implementing the SAME route paths,
status codes, JWT+httpOnly-refresh-cookie auth mechanism, and
{"error":{"code":"VALIDATION_ERROR","details":...}} error shape verified
against the actual Suzume repository in Stage A -- backed by a plain
in-memory dict, not Postgres/Prisma. This exists SPECIFICALLY so
SuzumeTrafficSource's real HTTP/auth code paths (including the
refresh-and-retry-on-401 logic) can be tested aggressively and safely,
without ever touching the real deployed application.

This is test infrastructure, not a production component -- it lives
under tests/, not src/agents/.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import jwt
from flask import Flask, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash

ACCESS_SECRET = "local-test-access-secret"
REFRESH_SECRET = "local-test-refresh-secret"
ACCESS_TTL_SECONDS = 900
REFRESH_TTL_SECONDS = 3600


def _now():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.isoformat()


def _validation_error(message: str, details=None):
    return jsonify({"error": {"message": message, "code": "VALIDATION_ERROR", "details": details or {}}}), 400


def _not_found(message="Resource not found"):
    return jsonify({"error": {"message": message, "code": "NOT_FOUND"}}), 404


def _unauthorized(message="Unauthorized"):
    return jsonify({"error": {"message": message, "code": "UNAUTHORIZED"}}), 401


def build_local_suzume_app() -> Flask:
    app = Flask("local_suzume_target")

    db = {
        "users": {}, "users_by_id": {}, "refresh_tokens": {}, "companies": {},
        "applications": {}, "rounds": {}, "experiences": {}, "questions": {},
        "learnings": {}, "action_items": {}, "prep_topics": {}, "prep_logs": {}, "prep_sources": {},
    }

    def issue_tokens(user):
        access = jwt.encode({"userId": user["id"], "email": user["email"], "exp": _now() + timedelta(seconds=ACCESS_TTL_SECONDS)}, ACCESS_SECRET, algorithm="HS256")
        refresh = jwt.encode({"userId": user["id"], "exp": _now() + timedelta(seconds=REFRESH_TTL_SECONDS)}, REFRESH_SECRET, algorithm="HS256")
        db["refresh_tokens"][refresh] = {"user_id": user["id"], "revoked": False}
        return access, refresh

    def require_auth():
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return None
        token = header[len("Bearer "):]
        try:
            return jwt.decode(token, ACCESS_SECRET, algorithms=["HS256"])
        except jwt.PyJWTError:
            return None

    def public_user(user):
        return {"id": user["id"], "name": user["name"], "email": user["email"], "createdAt": user["createdAt"], "updatedAt": user["updatedAt"]}

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/auth/register")
    def register():
        body = request.get_json(silent=True) or {}
        email = (body.get("email") or "").strip().lower()
        name = (body.get("name") or "").strip()
        password = body.get("password") or ""
        if not email or "@" not in email:
            return _validation_error("Enter a valid email address")
        if len(name) < 2:
            return _validation_error("Name must be at least 2 characters")
        if len(password) < 8:
            return _validation_error("Password must be at least 8 characters")
        if email in db["users"]:
            return jsonify({"error": {"message": "An account with this email already exists", "code": "CONFLICT"}}), 409
        user = {"id": str(uuid.uuid4()), "name": name, "email": email, "password_hash": generate_password_hash(password),
                "createdAt": _iso(_now()), "updatedAt": _iso(_now())}
        db["users"][email] = user
        db["users_by_id"][user["id"]] = user
        access, refresh = issue_tokens(user)
        resp = jsonify({"user": public_user(user), "accessToken": access, "needsPreparationSetup": True})
        resp.set_cookie("suzume_refresh_token", refresh, httponly=True, path="/api/auth")
        return resp, 201

    @app.post("/api/auth/login")
    def login():
        body = request.get_json(silent=True) or {}
        email = (body.get("email") or "").strip().lower()
        password = body.get("password") or ""
        user = db["users"].get(email)
        if not user or not check_password_hash(user["password_hash"], password):
            return _unauthorized("Invalid email or password")
        access, refresh = issue_tokens(user)
        resp = jsonify({"user": public_user(user), "accessToken": access, "needsPreparationSetup": False})
        resp.set_cookie("suzume_refresh_token", refresh, httponly=True, path="/api/auth")
        return resp, 200

    @app.post("/api/auth/refresh")
    def refresh():
        token = request.cookies.get("suzume_refresh_token") or (request.get_json(silent=True) or {}).get("refreshToken")
        if not token:
            return _unauthorized("Refresh token is no longer valid")
        try:
            payload = jwt.decode(token, REFRESH_SECRET, algorithms=["HS256"])
        except jwt.PyJWTError:
            return _unauthorized("Invalid or expired refresh token")
        stored = db["refresh_tokens"].get(token)
        if not stored or stored["revoked"]:
            return _unauthorized("Refresh token is no longer valid")
        user = db["users_by_id"].get(payload["userId"])
        if not user:
            return _unauthorized("User no longer exists")
        stored["revoked"] = True
        access, new_refresh = issue_tokens(user)
        resp = jsonify({"user": public_user(user), "accessToken": access})
        resp.set_cookie("suzume_refresh_token", new_refresh, httponly=True, path="/api/auth")
        return resp, 200

    @app.post("/api/auth/logout")
    def logout():
        token = request.cookies.get("suzume_refresh_token")
        if token and token in db["refresh_tokens"]:
            db["refresh_tokens"][token]["revoked"] = True
        resp = app.response_class(status=204)
        resp.set_cookie("suzume_refresh_token", "", expires=0, path="/api/auth")
        return resp

    @app.get("/api/auth/me")
    def me():
        payload = require_auth()
        if not payload:
            return _unauthorized("Missing or invalid authorization header")
        user = db["users_by_id"].get(payload["userId"])
        if not user:
            return _unauthorized()
        return jsonify({"user": public_user(user), "needsPreparationSetup": False})

    @app.get("/api/companies")
    def list_companies():
        if not require_auth():
            return _unauthorized()
        return jsonify({"companies": list(db["companies"].values())})

    @app.post("/api/companies")
    def create_company():
        if not require_auth():
            return _unauthorized()
        body = request.get_json(silent=True) or {}
        name = (body.get("name") or "").strip()
        if not name:
            return _validation_error("Company name is required")
        company = {"id": str(uuid.uuid4()), "name": name, "website": body.get("website")}
        db["companies"][company["id"]] = company
        return jsonify({"company": company}), 201

    @app.get("/api/companies/<cid>")
    def get_company(cid):
        if not require_auth():
            return _unauthorized()
        company = db["companies"].get(cid)
        if not company:
            return _not_found("Company not found")
        return jsonify({"company": company})

    def _find_or_create_company(name, website=None):
        for c in db["companies"].values():
            if c["name"] == name:
                return c
        company = {"id": str(uuid.uuid4()), "name": name, "website": website}
        db["companies"][company["id"]] = company
        return company

    VALID_STATUSES = {"INTERESTED", "APPLIED", "SHORTLISTED", "ASSESSMENT", "INTERVIEW", "OFFER", "REJECTED", "WITHDRAWN"}

    @app.get("/api/applications")
    def list_applications():
        payload = require_auth()
        if not payload:
            return _unauthorized()
        status = request.args.get("status")
        apps = [a for a in db["applications"].values() if a["userId"] == payload["userId"] and (not status or a["status"] == status)]
        return jsonify({"applications": apps})

    @app.post("/api/applications")
    def create_application():
        payload = require_auth()
        if not payload:
            return _unauthorized()
        body = request.get_json(silent=True) or {}
        company_name = (body.get("companyName") or "").strip()
        role = (body.get("role") or "").strip()
        if not company_name:
            return _validation_error("Company name is required")
        if not role:
            return _validation_error("Role is required")
        status = body.get("status", "INTERESTED")
        if status not in VALID_STATUSES:
            return _validation_error("Invalid status", {"status": ["invalid enum value"]})
        if "internship" in body and not isinstance(body["internship"], bool):
            return _validation_error("internship must be a boolean", {"internship": ["expected boolean"]})
        company = _find_or_create_company(company_name, body.get("companyWebsite"))
        application = {
            "id": str(uuid.uuid4()), "userId": payload["userId"], "companyId": company["id"], "company": company,
            "role": role, "location": body.get("location"), "status": status,
            "internship": bool(body.get("internship", False)), "rounds": [],
            "createdAt": _iso(_now()), "updatedAt": _iso(_now()),
        }
        db["applications"][application["id"]] = application
        return jsonify({"application": application}), 201

    @app.get("/api/applications/<aid>")
    def get_application(aid):
        payload = require_auth()
        if not payload:
            return _unauthorized()
        application = db["applications"].get(aid)
        if not application or application["userId"] != payload["userId"]:
            return _not_found("Application not found")
        return jsonify({"application": application})

    @app.patch("/api/applications/<aid>")
    def update_application(aid):
        payload = require_auth()
        if not payload:
            return _unauthorized()
        application = db["applications"].get(aid)
        if not application or application["userId"] != payload["userId"]:
            return _not_found("Application not found")
        body = request.get_json(silent=True) or {}
        if "status" in body and body["status"] not in VALID_STATUSES:
            return _validation_error("Invalid status")
        application.update({k: v for k, v in body.items() if k in ("role", "location", "status", "notes")})
        application["updatedAt"] = _iso(_now())
        return jsonify({"application": application})

    @app.delete("/api/applications/<aid>")
    def delete_application(aid):
        payload = require_auth()
        if not payload:
            return _unauthorized()
        application = db["applications"].get(aid)
        if not application or application["userId"] != payload["userId"]:
            return _not_found("Application not found")
        del db["applications"][aid]
        return "", 204

    @app.get("/api/applications/<aid>/rounds")
    def list_rounds(aid):
        if not require_auth():
            return _unauthorized()
        rounds = [r for r in db["rounds"].values() if r["applicationId"] == aid]
        return jsonify({"rounds": rounds})

    @app.post("/api/applications/<aid>/rounds")
    def create_round(aid):
        if not require_auth():
            return _unauthorized()
        application = db["applications"].get(aid)
        if not application:
            return _not_found("Application not found")
        body = request.get_json(silent=True) or {}
        title = (body.get("title") or "").strip()
        rtype = body.get("type")
        if not title:
            return _validation_error("Title is required")
        if not rtype:
            return _validation_error("Round type is required")
        round_ = {"id": str(uuid.uuid4()), "applicationId": aid, "type": rtype, "title": title,
                  "status": body.get("status", "UPCOMING"), "createdAt": _iso(_now())}
        db["rounds"][round_["id"]] = round_
        return jsonify({"round": round_}), 201

    @app.get("/api/rounds/<rid>")
    def get_round(rid):
        if not require_auth():
            return _unauthorized()
        round_ = db["rounds"].get(rid)
        if not round_:
            return _not_found("Round not found")
        return jsonify({"round": round_})

    @app.patch("/api/rounds/<rid>")
    def update_round(rid):
        if not require_auth():
            return _unauthorized()
        round_ = db["rounds"].get(rid)
        if not round_:
            return _not_found("Round not found")
        body = request.get_json(silent=True) or {}
        round_.update({k: v for k, v in body.items() if k in ("title", "status", "notes")})
        return jsonify({"round": round_})

    @app.delete("/api/rounds/<rid>")
    def delete_round(rid):
        if not require_auth():
            return _unauthorized()
        if rid not in db["rounds"]:
            return _not_found("Round not found")
        del db["rounds"][rid]
        return "", 204

    @app.post("/api/rounds/<rid>/experience")
    def create_experience(rid):
        if not require_auth():
            return _unauthorized()
        if rid not in db["rounds"]:
            return _not_found("Round not found")
        body = request.get_json(silent=True) or {}
        confidence = body.get("confidence")
        if confidence is not None and not (1 <= confidence <= 10):
            return _validation_error("confidence must be between 1 and 10")
        experience = {"id": str(uuid.uuid4()), "roundId": rid, "summary": body.get("summary"),
                      "confidence": confidence, "topicsCovered": body.get("topicsCovered", []), "questions": []}
        db["experiences"][experience["id"]] = experience
        return jsonify({"experience": experience}), 201

    @app.get("/api/experiences")
    def list_experiences():
        if not require_auth():
            return _unauthorized()
        return jsonify({"experiences": list(db["experiences"].values())})

    @app.get("/api/experiences/<eid>")
    def get_experience(eid):
        if not require_auth():
            return _unauthorized()
        experience = db["experiences"].get(eid)
        if not experience:
            return _not_found("Experience not found")
        return jsonify({"experience": experience})

    @app.patch("/api/experiences/<eid>")
    def update_experience(eid):
        if not require_auth():
            return _unauthorized()
        experience = db["experiences"].get(eid)
        if not experience:
            return _not_found("Experience not found")
        body = request.get_json(silent=True) or {}
        experience.update({k: v for k, v in body.items() if k in ("summary", "confidence", "topicsCovered")})
        return jsonify({"experience": experience})

    @app.post("/api/experiences/<eid>/questions")
    def create_question(eid):
        if not require_auth():
            return _unauthorized()
        if eid not in db["experiences"]:
            return _not_found("Experience not found")
        body = request.get_json(silent=True) or {}
        text = (body.get("question") or "").strip()
        category = body.get("category")
        valid_categories = {"DSA", "DBMS", "SQL", "OS", "OOP", "SYSTEM_DESIGN", "PROJECTS", "BEHAVIORAL", "OTHER"}
        if not text:
            return _validation_error("Question text is required")
        if category not in valid_categories:
            return _validation_error("Invalid category")
        question = {"id": str(uuid.uuid4()), "experienceId": eid, "question": text, "category": category}
        db["questions"][question["id"]] = question
        return jsonify({"question": question}), 201

    @app.patch("/api/questions/<qid>")
    def update_question(qid):
        if not require_auth():
            return _unauthorized()
        question = db["questions"].get(qid)
        if not question:
            return _not_found("Question not found")
        body = request.get_json(silent=True) or {}
        question.update({k: v for k, v in body.items() if k in ("question", "notes", "difficulty", "performance")})
        return jsonify({"question": question})

    @app.delete("/api/questions/<qid>")
    def delete_question(qid):
        if not require_auth():
            return _unauthorized()
        if qid not in db["questions"]:
            return _not_found("Question not found")
        del db["questions"][qid]
        return "", 204

    @app.get("/api/learnings")
    def list_learnings():
        if not require_auth():
            return _unauthorized()
        return jsonify({"learnings": list(db["learnings"].values())})

    @app.post("/api/learnings")
    def create_learning():
        if not require_auth():
            return _unauthorized()
        body = request.get_json(silent=True) or {}
        title = (body.get("title") or "").strip()
        category = body.get("category")
        valid_categories = {"COMMUNICATION", "TECHNICAL", "TIME_MANAGEMENT", "PROBLEM_SOLVING", "BEHAVIORAL", "OTHER"}
        if not title:
            return _validation_error("Title is required")
        if category not in valid_categories:
            return _validation_error("Invalid category")
        learning = {"id": str(uuid.uuid4()), "title": title, "category": category,
                    "priority": body.get("priority", "MEDIUM"), "status": body.get("status", "OPEN"), "actionItems": []}
        db["learnings"][learning["id"]] = learning
        return jsonify({"learning": learning}), 201

    @app.get("/api/learnings/<lid>")
    def get_learning(lid):
        if not require_auth():
            return _unauthorized()
        learning = db["learnings"].get(lid)
        if not learning:
            return _not_found("Learning not found")
        return jsonify({"learning": learning})

    @app.patch("/api/learnings/<lid>")
    def update_learning(lid):
        if not require_auth():
            return _unauthorized()
        learning = db["learnings"].get(lid)
        if not learning:
            return _not_found("Learning not found")
        body = request.get_json(silent=True) or {}
        learning.update({k: v for k, v in body.items() if k in ("title", "description", "priority", "status")})
        return jsonify({"learning": learning})

    @app.delete("/api/learnings/<lid>")
    def delete_learning(lid):
        if not require_auth():
            return _unauthorized()
        if lid not in db["learnings"]:
            return _not_found("Learning not found")
        del db["learnings"][lid]
        return "", 204

    @app.post("/api/learnings/<lid>/actions")
    def create_action_item(lid):
        if not require_auth():
            return _unauthorized()
        if lid not in db["learnings"]:
            return _not_found("Learning not found")
        body = request.get_json(silent=True) or {}
        title = (body.get("title") or "").strip()
        if not title:
            return _validation_error("Title is required")
        item = {"id": str(uuid.uuid4()), "learningId": lid, "title": title, "status": body.get("status", "PENDING")}
        db["action_items"][item["id"]] = item
        return jsonify({"actionItem": item}), 201

    @app.patch("/api/actions/<aid>")
    def update_action_item(aid):
        if not require_auth():
            return _unauthorized()
        item = db["action_items"].get(aid)
        if not item:
            return _not_found("Action item not found")
        body = request.get_json(silent=True) or {}
        item.update({k: v for k, v in body.items() if k in ("title", "status", "description")})
        return jsonify({"actionItem": item})

    @app.delete("/api/actions/<aid>")
    def delete_action_item(aid):
        if not require_auth():
            return _unauthorized()
        if aid not in db["action_items"]:
            return _not_found("Action item not found")
        del db["action_items"][aid]
        return "", 204

    @app.get("/api/preparation")
    def list_preparation():
        if not require_auth():
            return _unauthorized()
        return jsonify({"topics": list(db["prep_topics"].values())})

    @app.get("/api/preparation/activity")
    def preparation_activity():
        if not require_auth():
            return _unauthorized()
        return jsonify({"activity": []})

    @app.post("/api/preparation/setup")
    def preparation_setup():
        if not require_auth():
            return _unauthorized()
        return jsonify({"status": "completed"})

    @app.post("/api/preparation/topics")
    def create_preparation_topic():
        if not require_auth():
            return _unauthorized()
        body = request.get_json(silent=True) or {}
        name = (body.get("name") or "").strip()
        if not name:
            return _validation_error("Topic name is required")
        topic = {"id": str(uuid.uuid4()), "name": name, "category": body.get("category", "")}
        db["prep_topics"][topic["id"]] = topic
        return jsonify({"topic": topic}), 201

    @app.delete("/api/preparation/topics/<tid>")
    def delete_preparation_topic(tid):
        if not require_auth():
            return _unauthorized()
        if tid not in db["prep_topics"]:
            return _not_found("Topic not found")
        del db["prep_topics"][tid]
        return "", 204

    @app.get("/api/preparation/logs")
    def list_preparation_logs():
        if not require_auth():
            return _unauthorized()
        return jsonify({"logs": list(db["prep_logs"].values())})

    @app.post("/api/preparation/logs")
    def create_preparation_log():
        if not require_auth():
            return _unauthorized()
        body = request.get_json(silent=True) or {}
        if not body.get("date"):
            return _validation_error("date is required")
        log = {"id": str(uuid.uuid4()), "date": body.get("date"), "questionsSolved": body.get("questionsSolved", 0)}
        db["prep_logs"][log["id"]] = log
        return jsonify({"log": log}), 201

    @app.patch("/api/preparation/logs/<lid>")
    def update_preparation_log(lid):
        if not require_auth():
            return _unauthorized()
        log = db["prep_logs"].get(lid)
        if not log:
            return _not_found("Log not found")
        body = request.get_json(silent=True) or {}
        log.update({k: v for k, v in body.items() if k in ("questionsSolved", "durationMinutes", "notes")})
        return jsonify({"log": log})

    @app.delete("/api/preparation/logs/<lid>")
    def delete_preparation_log(lid):
        if not require_auth():
            return _unauthorized()
        if lid not in db["prep_logs"]:
            return _not_found("Log not found")
        del db["prep_logs"][lid]
        return "", 204

    @app.get("/api/preparation/sources")
    def list_preparation_sources():
        if not require_auth():
            return _unauthorized()
        return jsonify({"sources": list(db["prep_sources"].values())})

    @app.post("/api/preparation/sources")
    def create_preparation_source():
        if not require_auth():
            return _unauthorized()
        body = request.get_json(silent=True) or {}
        if not body.get("profileUrl"):
            return _validation_error("profileUrl is required")
        source = {"id": str(uuid.uuid4()), "provider": body.get("provider"), "profileUrl": body.get("profileUrl")}
        db["prep_sources"][source["id"]] = source
        return jsonify({"source": source}), 201

    @app.post("/api/preparation/sources/<sid>/refresh")
    def refresh_preparation_source(sid):
        if not require_auth():
            return _unauthorized()
        if sid not in db["prep_sources"]:
            return _not_found("Source not found")
        return jsonify({"status": "refreshed"})

    @app.delete("/api/preparation/sources/<sid>")
    def delete_preparation_source(sid):
        if not require_auth():
            return _unauthorized()
        if sid not in db["prep_sources"]:
            return _not_found("Source not found")
        del db["prep_sources"][sid]
        return "", 204

    @app.patch("/api/preparation/<tid>")
    def update_preparation_topic_progress(tid):
        if not require_auth():
            return _unauthorized()
        return jsonify({"topicId": tid, "updated": True})

    @app.get("/api/dashboard/summary")
    def dashboard_summary():
        if not require_auth():
            return _unauthorized()
        return jsonify({"activeApplications": 0, "upcoming": [], "topics": []})

    @app.get("/api/calendar/events")
    def calendar_events():
        if not require_auth():
            return _unauthorized()
        return jsonify({"events": []})

    @app.get("/api/analytics/overview")
    def analytics_overview():
        if not require_auth():
            return _unauthorized()
        return jsonify({"applications": 0, "interviews": 0})

    @app.post("/api/extraction/parse")
    def extraction_parse():
        if not require_auth():
            return _unauthorized()
        body = request.get_json(silent=True) or {}
        text = body.get("text", "")
        if not text or not text.strip():
            return _validation_error("text is required")
        return jsonify({"extracted": {"company": None, "role": None, "confidence": 0.0}})

    @app.errorhandler(404)
    def handle_404(_e):
        return jsonify({"error": {"message": "Route not found", "code": "NOT_FOUND"}}), 404

    return app
