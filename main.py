#!/usr/bin/env python
"""
Render startup script for CrowdSense AI Backend
This wrapper allows Render to easily locate and run the FastAPI app
"""
import os
import sys
import uvicorn

if __name__ == "__main__":
    # Add the current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    
    # Run the FastAPI app from server.main module
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
