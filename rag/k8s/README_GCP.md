# Cloud Setup Commands Summary

## 1. Build and Push Image
docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT_ID/fhir-rag-repo/rag-api:v1 .
docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/fhir-rag-repo/rag-api:v1

## 2. Setup Kubernetes Secrets
kubectl create secret generic rag-secrets \
    --from-literal=openrouter-key=YOUR_OPENROUTER_KEY

## 3. Apply Kubernetes Manifests
kubectl apply -f k8s/deployment.yaml

## 4. Verify Pods
kubectl get pods
kubectl logs -f deployment/rag-api -c rag-api

## 5. Get Internal Load Balancer IP
kubectl get svc rag-api-ilb
