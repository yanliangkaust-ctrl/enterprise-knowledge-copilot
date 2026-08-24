# OCR Platform API Specification

## Overview
The API Gateway is the public interface for the Government OCR Platform. It accepts document jobs and exposes status and result endpoints for authorized consumers.

## Endpoints
- `POST /v1/ocr/jobs` accepts a document reference from Document Storage and returns a job identifier.
- `GET /v1/ocr/jobs/{job_id}` returns processing status from the OCR Service.
- `GET /v1/ocr/jobs/{job_id}/result` returns the approved OCR output.

## Authentication and Access
Requests use the government identity provider and service-to-service tokens. The Security Team owns the authentication review. The API Gateway must reject missing scopes and record correlation IDs for the Monitoring Service.

## Reliability Requirements
The API Gateway should return a clear error when the OCR Service is unavailable. Clients must use an idempotency key for retries. Job status remains available during OCR processing so the Operations Team can investigate incidents.

## Dependencies
The API depends on the OCR Service, Document Storage, Kubernetes networking, and the Monitoring Service. Contract changes require UAT Environment validation before production deployment.
