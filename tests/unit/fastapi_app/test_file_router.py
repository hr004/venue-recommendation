from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from src.main import create_app


class TestFileRouter:
    """Test class for the file_router API endpoints."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, monkeypatch):
        """Setup mocks for OpenAI clients before each test."""
        # Mock OpenAI clients to prevent real API calls
        mock_chat = MagicMock()
        mock_embeddings = MagicMock()
        monkeypatch.setattr("src.agents.base_agent.ChatOpenAI", mock_chat)
        monkeypatch.setattr("src.engine.vector_store.OpenAIEmbeddings", mock_embeddings)

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def mock_current_requests_db(self):
        """Mock current requests database."""
        return {
            "EVT-2026-001": {
                "event_id": "EVT-2026-001",
                "client_id": "CLI-158",
                "client_name": "Test Client",
                "attendee_count": 100,
                "location_requirements": {"cities": ["New York", "Boston"]},
            },
            "EVT-2026-002": {
                "event_id": "EVT-2026-002",
                "client_id": "CLI-159",
                "client_name": "Another Client",
                "attendee_count": 50,
                "location_requirements": {"cities": ["San Francisco"]},
            },
        }

    @pytest.fixture
    def mock_venue_recommendations_response(self):
        """Mock venue recommendations response."""
        return {
            "recommendations": [
                {
                    "venue_id": "VEN-001",
                    "venue_name": "Test Venue 1",
                    "ranking": 1,
                    "estimated_cost": 5000,
                    "analysis": {
                        "capacity_agent": "Capacity analysis for venue 1",
                        "amenity_agent": "Amenity analysis for venue 1",
                        "location_agent": "Location analysis for venue 1",
                        "cost_agent": "Cost analysis for venue 1",
                        "similar_events": "Similar events analysis for venue 1",
                    },
                    "strengths": ["Great location", "Spacious"],
                    "considerations": ["Limited parking"],
                },
                {
                    "venue_id": "VEN-002",
                    "venue_name": "Test Venue 2",
                    "ranking": 2,
                    "estimated_cost": 4000,
                    "analysis": {
                        "capacity_agent": "Capacity analysis for venue 2",
                        "amenity_agent": "Amenity analysis for venue 2",
                        "location_agent": "Location analysis for venue 2",
                        "cost_agent": "Cost analysis for venue 2",
                        "similar_events": "Similar events analysis for venue 2",
                    },
                    "strengths": ["Affordable", "Modern facilities"],
                    "considerations": ["Smaller capacity"],
                },
            ]
        }

    @pytest.fixture
    def mock_retrieved_documents(self):
        """Mock retrieved documents from vector search."""
        return [
            {
                "metadata": {
                    "event": {"event_id": "EVT-2025-001"},
                    "venue": {"venue_id": "VEN-001", "venue_name": "Test Venue 1"},
                    "client": {"client_id": "CLI-158"},
                },
                "text": "Sample event-venue match text",
            }
        ]

    @patch("src.fastapi_app.file_router.get_current_requests_db")
    @patch("src.fastapi_app.file_router.VenueRecommendationOrchestrator")
    @patch("src.fastapi_app.file_router.OpenSearchEngine")
    def test_get_venue_recommendations_success(
        self,
        mock_opensearch_engine,
        mock_orchestrator,
        mock_get_db,
        client,
        mock_current_requests_db,
        mock_retrieved_documents,
        mock_venue_recommendations_response,
    ):
        """Test successful venue recommendations request."""
        mock_get_db.return_value = mock_current_requests_db

        mock_engine_instance = MagicMock()
        mock_engine_instance.search_documents = AsyncMock(
            return_value=mock_retrieved_documents
        )
        mock_opensearch_engine.return_value = mock_engine_instance

        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_instance.recommend = AsyncMock(
            return_value=mock_venue_recommendations_response
        )
        mock_orchestrator.return_value = mock_orchestrator_instance

        request_data = {"event_id": "EVT-2026-001", "top_n": 3}
        response = client.post("/api/v1/venues/recommend", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) == 2
        assert data["recommendations"][0]["venue_id"] == "VEN-001"
        assert data["recommendations"][0]["ranking"] == 1

        mock_engine_instance.search_documents.assert_called_once()
        mock_orchestrator_instance.recommend.assert_called_once()

    @patch("src.fastapi_app.file_router.get_current_requests_db")
    @patch("src.fastapi_app.file_router.OpenSearchEngine")
    @patch("src.fastapi_app.file_router.VenueRecommendationOrchestrator")
    def test_get_venue_recommendations_event_not_found(
        self,
        mock_orchestrator,
        mock_opensearch_engine,
        mock_get_db,
        client,
        mock_current_requests_db,
    ):
        """Test venue recommendations request with non-existent event ID."""
        mock_get_db.return_value = mock_current_requests_db

        mock_engine_instance = MagicMock()
        mock_engine_instance.search_documents = AsyncMock()
        mock_opensearch_engine.return_value = mock_engine_instance

        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_instance.recommend = AsyncMock()
        mock_orchestrator.return_value = mock_orchestrator_instance

        request_data = {"event_id": "EVT-NONEXISTENT", "top_n": 3}
        response = client.post("/api/v1/venues/recommend", json=request_data)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert "vent not in current requests database" in data["detail"]

    @patch("src.fastapi_app.file_router.get_current_requests_db")
    @patch("src.fastapi_app.file_router.OpenSearchEngine")
    @patch("src.fastapi_app.file_router.VenueRecommendationOrchestrator")
    def test_get_venue_recommendations_invalid_request(
        self,
        mock_orchestrator,
        mock_opensearch_engine,
        mock_get_db,
        client,
        mock_current_requests_db,
    ):
        """Test venue recommendations request with invalid request data."""
        mock_get_db.return_value = mock_current_requests_db

        mock_engine_instance = MagicMock()
        mock_engine_instance.search_documents = AsyncMock()
        mock_opensearch_engine.return_value = mock_engine_instance

        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_instance.recommend = AsyncMock()
        mock_orchestrator.return_value = mock_orchestrator_instance

        request_data = {
            "event_id": "EVT-2026-001",
            "top_n": ["invalid"],  # Invalid type - should cause validation error
        }
        response = client.post("/api/v1/venues/recommend", json=request_data)

        assert response.status_code == 422
        assert "detail" in response.json()

    @patch("src.fastapi_app.file_router.OpenSearchEngine")
    def test_index_documents_success(self, mock_opensearch_engine, client):
        """Test successful document indexing."""
        mock_engine_instance = MagicMock()
        mock_engine_instance.index_documents = AsyncMock(return_value=(True, 10))
        mock_opensearch_engine.return_value = mock_engine_instance

        request_data = {"event_history_path": "data/venue/event_history.json"}
        response = client.post("/api/v1/index", json=request_data)

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["total_documents"] == 10

        mock_engine_instance.index_documents.assert_called_once_with(
            "data/venue/event_history.json"
        )

    @patch("src.fastapi_app.file_router.OpenSearchEngine")
    def test_index_documents_failure(self, mock_opensearch_engine, client):
        """Test document indexing failure."""
        mock_engine_instance = MagicMock()
        mock_engine_instance.index_documents = AsyncMock(return_value=(False, 0))
        mock_opensearch_engine.return_value = mock_engine_instance

        request_data = {"event_history_path": "data/venue/event_history.json"}
        response = client.post("/api/v1/index", json=request_data)

        data = response.json()
        assert data["success"] is False
        assert data["total_documents"] == 0

    @patch("src.fastapi_app.file_router.OpenSearchEngine")
    def test_index_documents_invalid_request(self, mock_opensearch_engine, client):
        """Test document indexing with invalid request data."""
        mock_engine_instance = MagicMock()
        mock_engine_instance.index_documents = AsyncMock(return_value=(False, 0))
        mock_opensearch_engine.return_value = mock_engine_instance

        request_data = {
            "event_history_path": [
                "invalid"
            ]  # Invalid type - should cause validation error
        }
        response = client.post("/api/v1/index", json=request_data)

        assert response.status_code == 422
        assert "detail" in response.json()
