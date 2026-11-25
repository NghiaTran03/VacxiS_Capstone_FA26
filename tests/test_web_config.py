"""Verify that web selects the main architecture without changing CLI baselines."""

from unittest.mock import MagicMock, Mock

import pytest

from vacxis_capstone_fa26 import cli, web


def test_web_defaults_match_combined_table_aware_lightrag_architecture():
    config = cli.retrieval_config(web.parse_args([]))
    assert config.mode == "hybrid_lightrag"
    assert config.collection_name == "VacxiS_Capstone_FA26_table_dual"
    assert config.lightrag_mode == "hybrid"
    assert config.lightrag_fusion == "rrf_rerank"
    assert config.lightrag_rerank_weight == pytest.approx(0.85)
    assert config.lightrag_top_k == 30
    assert config.lightrag_chunk_top_k == 20
    assert config.lightrag_max_entity_tokens == 3000
    assert config.lightrag_max_relation_tokens == 3000
    assert config.lightrag_max_total_tokens == 30000
    assert not config.skip_rerank


def test_explicit_web_arguments_still_override_defaults():
    config = cli.retrieval_config(web.parse_args([
        "--retriever-mode", "hybrid", "--collection-name", "existing_collection",
        "--lightrag-mode", "local", "--lightrag-fusion", "context_append",
    ]))
    assert config.mode == "hybrid"
    assert config.collection_name == "existing_collection"
    assert config.lightrag_mode == "local"
    assert config.lightrag_fusion == "context_append"


def test_web_standalone_lightrag_uses_lightrag_package_defaults():
    config = cli.retrieval_config(web.parse_args(["--retriever-mode", "lightrag"]))
    assert config.mode == "lightrag"
    assert config.lightrag_mode == "mix"
    assert config.lightrag_top_k == 40
    assert config.lightrag_chunk_top_k == 20
    assert config.lightrag_max_entity_tokens == 6000
    assert config.lightrag_max_relation_tokens == 8000
    assert config.lightrag_max_total_tokens == 30000
    assert not config.skip_rerank


def test_web_defaults_do_not_change_workflow_cli_baseline():
    assert cli.parse_args(["ask", "--query", "Verorab?"]).retriever_mode == "hybrid"


def test_empty_preview_corpus_reports_a_clean_cli_error(monkeypatch):
    monkeypatch.setattr(web, "Documents", Mock(return_value=Mock(indexed_ids=set())))
    with pytest.raises(SystemExit, match="Preview requires an indexed --chunks file"):
        web.main(["--preview"])


def test_web_passes_main_architecture_to_real_turn_and_logs_settings(monkeypatch, capsys):
    pipeline = MagicMock()
    pipeline.__enter__.return_value = pipeline
    pipeline.run_turn.return_value = "answer"
    pipeline_factory = Mock(return_value=pipeline)
    planner = Mock()
    monkeypatch.setattr(web, "VacxisRAGPipeline", pipeline_factory)
    monkeypatch.setattr(web, "QueryPlanner", Mock(return_value=planner))
    monkeypatch.setattr(web, "validate_config", Mock(return_value="model"))
    monkeypatch.setattr(web, "Documents", Mock())
    monkeypatch.setattr(web, "create_server", Mock())
    monkeypatch.setattr("vacxis_capstone_fa26.web.sharing.run_server", Mock())

    def build(run_turn, documents, *, preview):
        assert not preview
        assert run_turn("question") == "answer"
        return Mock()

    monkeypatch.setattr(web, "build_app", build)
    assert web.main([]) == 0
    config = pipeline_factory.call_args.kwargs["retrieval_config"]
    assert config.mode == "hybrid_lightrag"
    assert config.collection_name == "VacxiS_Capstone_FA26_table_dual"
    assert config.lightrag_fusion == "rrf_rerank"
    assert pipeline_factory.call_args.kwargs["query_planner"] is planner
    assert "Retriever: hybrid_lightrag" in capsys.readouterr().out
    pipeline.__exit__.assert_called_once()
