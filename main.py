from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Add src to path to import existing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from backend.routes import physics, ai, simulation

app = FastAPI(title="ESP Digital Twin API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(physics.router, prefix="/api/physics", tags=["Physics"])
app.include_router(ai.router, prefix="/api/ai", tags=["AI"])
app.include_router(simulation.router, prefix="/api/simulation", tags=["Simulation"])

@app.get("/")
async def root():
    return {"message": "ESP Digital Twin API is running"}
