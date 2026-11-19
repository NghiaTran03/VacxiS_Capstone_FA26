import json
from unittest.mock import Mock

import pytest

from vacxis_capstone_fa26.generation.cli import main


def test_cli_ignores_planner_env_for_online_queries(monkeypatch, capsys):
    from vacxis_capstone_fa26.generation import cli
    from vacxis_capstone_fa26.retrieval import retriever
    from vacxis_capstone_fa26.generation.schemas import GeneratedAnswer
    monkeypatch.setenv("VACXIS_QUERY_PLANNER_MODEL", "planner-from-env")
    monkeypatch.setenv("VACXIS_GENERATION_MODEL", "generation-from-env")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    monkeypatch.setattr("sys.argv", ["vacxis-generation", "--query", "Verorab là gì?", "--json"])
    monkeypatch.setattr(retriever, "VacxisRetriever", Mock(return_value=Mock(
        __enter__=Mock(), __exit__=Mock(return_value=False),
    )))
    pipeline = Mock()
    pipeline.answer.return_value = GeneratedAnswer(query="Verorab là gì?", answer="Answer")
    factory = Mock(return_value=pipeline)
    monkeypatch.setattr(cli, "VacxisRAGPipeline", factory)
    main()
    assert "query_planner" not in factory.call_args.kwargs
    assert factory.call_args.kwargs["generator"].config.model == "generation-from-env"
    pipeline.run_turn.assert_not_called()
    assert json.loads(capsys.readouterr().out)["answer"] == "Answer"


@pytest.mark.parametrize("arguments", [
    ["--query-planner-model", "fixture", "--dry-run"],
    ["--query-planner-model", "fixture", "--input-file", "test_dataset.json"],
    ["--clarification", "question", "answer"],
])
def test_cli_rejects_incompatible_modes(monkeypatch, arguments):
    monkeypatch.delenv("VACXIS_QUERY_PLANNER_MODEL", raising=False)
    monkeypatch.setattr("sys.argv", ["vacxis-generation", *arguments])
    with pytest.raises(SystemExit, match="2"):
        main()


@pytest.mark.parametrize("arguments", [
    ["--dry-run", "--query", "Verorab là gì?"],
    ["--input-file", "test_dataset.json"],
])
def test_cli_ignores_planner_env_outside_single_online_mode(monkeypatch, arguments):
    from vacxis_capstone_fa26.generation.cli import parse_args

    monkeypatch.setenv("VACXIS_QUERY_PLANNER_MODEL", "planner-from-env")
    monkeypatch.setattr("sys.argv", ["vacxis-generation", *arguments])

    assert not hasattr(parse_args(), "query_planner_model")
