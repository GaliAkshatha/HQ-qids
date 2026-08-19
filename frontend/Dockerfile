# NOTE (honesty, not a formality): not built or run in this project's
# development environment -- Docker itself was unavailable there
# (documented since Phase 7). Statically reviewed only.

FROM node:20-slim

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .

EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
