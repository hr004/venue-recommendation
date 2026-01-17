# Event Venue Recommendation System


## Table of Contents

- [Event Venue Recommendation System](#event-venue-recommendation-system)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Quick Start](#quick-start)
    - [Prerequisites](#prerequisites)
    - [Setup and Run](#setup-and-run)
  - [Architecture](#architecture)
    - [Core Components](#core-components)
    - [System Flow](#system-flow)
  - [API Endpoints](#api-endpoints)
    - [1. Get Venue Recommendations](#1-get-venue-recommendations)
    - [2. Index Documents](#2-index-documents)
  - [Configuration](#configuration)
  - [Key Features](#key-features)
    - [1. RAG Implementation](#1-rag-implementation)
  - [Documentation](#documentation)
  - [Project Structure](#project-structure)
  - [Development](#development)
    - [Running Tests](#running-tests)
    - [Debugging](#debugging)
  - [Troubleshooting](#troubleshooting)
    - [OpenSearch Connection Issues](#opensearch-connection-issues)
    - [API Errors](#api-errors)
    - [Agent Failures](#agent-failures)

## Overview

The Event Venue Recommendation System is a multi-agent AI system designed to match corporate event requirements. The system leverages Retrieval Augmented Generation (RAG) to find similar successful event-venue pairings from historical data and uses specialized AI agents to analyze different matching criteria.


## Quick Start

### Prerequisites

- Python 3.12+
- Docker and Docker Compose (for OpenSearch)
- OpenAI API key (for embeddings and text generation)
- `uv` package manager

### Setup and Run

Use `make` commands. Make sure you have `uv` installed, if not run `curl -LsSf https://astral.sh/uv/install.sh | sh`. Make sure to add OPENAI_API_KEY in `.env` file or `export OPENAI_API_KEY=<your api key>`.

- To run the server

```bash
# if python 3.12.x is not available
make isntall-python
# create venv
make create-venv
# install dependencies, start opensearch locally, create the index and run server
make run 
# index local document
make index-documents
```

Make a recommendation request

```
curl -X 'POST' \
  'http://0.0.0.0:8000/api/v1/venues/recommend' \
  -d '{
  "event_id": "EVT-2026-028",
  "top_n": 3
}'
```

The API will be available at `http://localhost:8000` and the api docs at `http://localhost:8000/api/docs`

- Execute `/index` endpoint to index `event_history.json` file
- Execute `/recommend` endpoint to get top-n recommendations for the venue


## Architecture

The system follows a multi-agent architecture with the following components:

### Core Components

1. **RAG Retriever** (`src/engine/vector_store.py`)
   - Vectorizes historical event data using OpenAI embeddings
   - Performs hybrid search (vector similarity + structured filters)
   - Filters by attendee count and location cities

2. **Specialized Agents** (`src/agents/`)
   - **Capacity & Space Agent**: Analyzes attendee fit, room layouts, breakout spaces
   - **Amenity Matching Agent**: Evaluates required equipment, services, catering
   - **Location Agent**: Assesses geographic preferences, accessibility, nearby accommodations
   - **Cost Analysis Agent**: Evaluates budget fit, value assessment, hidden costs

3. **Orchestrator** (`src/orchestrator.py`)
   - Coordinates all specialized agents
   - Executes agents in parallel using `asyncio.gather()`
   - Handles errors and retries failed agents (up to 3 times)
   - Aggregates results and passes to supervisor

4. **Supervisor Agent** (`src/agents/recommend.py`)
   - Synthesizes all agent analyses
   - Ranks venues based on combined scores
   - Generates final recommendations with strengths and considerations

5. **FastAPI Application** (`src/main.py`, `src/fastapi_app/`)
   - RESTful API endpoints for recommendations and indexing
   - Request/response validation using Pydantic models

### System Flow

```text
EventID → RAG Retriever → [4 Parallel Agents] → Supervisor → Recommendations
```

For a detailed system flow diagram, see [workflow.md](./workflow.md#system-flow).

## API Endpoints

### 1. Get Venue Recommendations

**Endpoint**: `POST /api/v1/venues/recommend`

**Request**:

```json
{
  "event_id": "EVT-2026-028",
  "top_n": 3
}
```

**Response**:

```json
{
  "recommendations": [
    {
      "venue_id": "VEN-442",
      "venue_name": "Pacific Convention Center",
      "ranking": 1,
      "estimated_cost": 68000,
      "analysis": {
        "capacity_agent": "...",
        "amenity_agent": "...",
        "location_agent": "...",
        "cost_agent": "...",
        "similar_events": "..."
      },
      "strengths": ["Proven track record", "All amenities in-house"],
      "considerations": ["Parking can be tight"]
    }
  ]
}
```

### 2. Index Documents

**Endpoint**: `POST /api/v1/index`

**Request**:

```json
{
  "event_history_path": "data/venue/event_history.json"
}
```

**Response**:

```json
{
  "success": true,
  "total_documents": 500
}
```

## Configuration

Configuration is managed in `src/config.py`. Key settings:

- **Server**: Host and port (default: `0.0.0.0:8000`)
- **OpenSearch**: URL and index name (default: `http://localhost:9200`, index: `venue_event_history`)
- **LLM**: Model and API settings (default: `gpt-4o-mini`)
- **Logger**: Logging level and structured logging settings

Environment variables can be loaded from `.env` file using `python-dotenv`.


## Key Features

### 1. RAG Implementation

- **Vectorization**: Historical events are embedded using OpenAI `text-embedding-3-small` (1536 dimensions)
- **Hybrid Search**: Combines vector similarity search with structured filters
- **Filtering**: Filters by attendee count (±20% tolerance) and location cities (keyword prefix search)
- **Metadata Enrichment**: Merges client profiles and venue details into document metadata


## Documentation

- **[Workflow Documentation](./docs/workflow.md)**: Detailed explanation of indexing, searching, and agent orchestration

## Project Structure

```text
event_venue_rec/
├── data/venue/              # Data files (events, venues, clients)
├── docs/                    # Documentation
│   ├── README.md           # This file
│   └── workflow.md         # Detailed workflow documentation
├── local/opensearch/        # OpenSearch configuration
│   ├── docker-compose.yml  # Docker setup for OpenSearch
│   ├── create_index.sh     # Index creation script
│   └── mappings.json       # OpenSearch index schema
├── src/
│   ├── agents/             # Specialized AI agents
│   │   ├── amenity.py      # Amenity matching agent
│   │   ├── capacity.py     # Capacity & space agent
│   │   ├── cost.py         # Cost analysis agent
│   │   ├── location.py     # Location agent
│   │   ├── recommend.py   # Supervisor/recommendation agent
│   │   └── base_agent.py   # Base agent class
│   ├── engine/
│   │   └── vector_store.py # OpenSearch vector store implementation
│   ├── fastapi_app/        # FastAPI routes and handlers
│   ├── datamodel/          # Pydantic models for API
│   ├── orchestrator.py     # Agent orchestration logic
│   ├── config.py           # Configuration management
│   └── main.py             # FastAPI application entry point
├── pyproject.toml          # Project dependencies
└── README.md               # Main project README
```

## Development

### Running Tests

This is a TODO

```bash
pytest tests
```

### Debugging

VS Code launch configuration is available in `.vscode/launch.json` for debugging the FastAPI server.

## Troubleshooting

### OpenSearch Connection Issues

- Ensure Docker containers are running: `docker ps`
- Check OpenSearch health: `curl http://localhost:9200/_cluster/health`
- Verify index exists: `curl http://localhost:9200/_cat/indices`

### API Errors

- Check that event_id exists in `current_requests.json`
- Verify OpenSearch index has been created and populated
- Check logs for detailed error messages

### Agent Failures

- Agents automatically retry up to 3 times
- Check logs for specific agent error messages
- Verify OpenAI API key is set correctly
