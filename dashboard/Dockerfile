# =============================================================================
# Artenic AI Dashboard — Production Dockerfile
# =============================================================================
# Build:  docker build -t artenic-ai-dashboard dashboard/
# Run:    docker run -p 80:80 artenic-ai-dashboard
# =============================================================================

# Stage 1: Build
FROM node:22-alpine AS builder

WORKDIR /app

# Copy package files (cache-friendly)
COPY dashboard/package*.json ./

# Install dependencies
RUN npm ci

# Copy source
COPY dashboard/ .

# Build for production (demo mode enabled by default)
ENV VITE_DEMO_MODE=true
RUN npm run build

# Stage 2: Production with Nginx
FROM nginx:alpine

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY dashboard/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://localhost/health || exit 1

CMD ["nginx", "-g", "daemon off;"]
