"""Vercel serverless entry point for FastAPI app."""

from main import app

# Vercel expects a variable named 'app' or 'handler'
# This file allows Vercel to find and serve the FastAPI app
