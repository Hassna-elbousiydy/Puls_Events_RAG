"""Les tests offline interdisent les connexions réseau."""
import socket
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    """Un appel API involontaire fait échouer le test."""
    def blocked(*args, **kwargs):
        raise AssertionError('Réseau interdit dans les tests offline.')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)


def pytest_collection_modifyitems(items):
    """Distingue les validations d'artefacts des unités autonomes."""
    for item in items:
        if item.path.name in {'test_preprocessing.py', 'test_chunking.py', 'test_vectorstore.py'}:
            item.add_marker(pytest.mark.artifacts)
            if not Path('data/processed/events_processed.csv').exists():
                item.add_marker(pytest.mark.skip(reason='Artefacts absents : reconstruire puis exécuter pytest -m artifacts.'))
