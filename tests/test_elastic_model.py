from unittest.mock import MagicMock, ANY, patch
import os
import importlib
import pytest
from elasticsearch import Elasticsearch
from pathlib import Path
import builtins

project_root = Path(__file__).parent.parent
fake_config_path = project_root / "example-conf.json"

class ElasticSetup:
    @staticmethod
    def setup():
        with patch.dict(os.environ, {"DOCUMENT_DEFINITION_CONFIG": str(fake_config_path)}), \
                patch.object(builtins, "open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "{\"identifier_field\": \"doc_id\", \"saved_fields\": {\"title\": \"text\", \"doc_id\": \"integer\", \"link\": \"text\", \"content\": \"text\"}, \"field_for_llm\": \"content\", \"model_name\": \"Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0\", \"field_to_embed\": \"content\"}"

            elastic_model_module = importlib.import_module("webiks_hebrew_ragbot.elastic_model")
        return (
            elastic_model_module.ElasticModel,
            elastic_model_module.index_from_doc_id,
            elastic_model_module.EMBEDDING_INDEX
        )

ElasticModel, index_from_doc_id, EMBEDDING_INDEX = ElasticSetup.setup()

def test_index_from_doc_id():
    doc_id = 1234
    expected_index = f"{EMBEDDING_INDEX}_1"
    assert index_from_doc_id(doc_id) == expected_index

def test_create_paragraph():
    es_mock = MagicMock(spec=Elasticsearch)
    model = ElasticModel(es_mock)
    paragraph = {"doc_id": 1, "content": "Some text content"}
    es_mock.index = MagicMock()
    model.create_paragraph(paragraph)
    index = index_from_doc_id(int(paragraph["doc_id"]))
    es_mock.index.assert_called_once_with(index=index, body={
        "last_update": ANY,
        **paragraph
    })

def test_create_documents_no_delete():
    es_mock = MagicMock(spec=Elasticsearch)
    model = ElasticModel(es_mock)
    es_mock.search.return_value = {"hits": {"hits": []}}
    new_doc = {"doc_id": 1, "title": "New Title", "content": "New Content"}
    model.create_or_update_documents([new_doc], update=False)
    es_mock.search.assert_not_called()
    es_mock.delete.assert_not_called()
    es_mock.index.assert_called_once()

def test_update_documents_with_delete():
    es_mock = MagicMock(spec=Elasticsearch)
    model = ElasticModel(es_mock)
    es_mock.search.return_value = {"hits": {"hits": [{"_id": "123", "doc_id": 1}]}}
    new_doc = {"doc_id": 1, "title": "Updated Title", "content": "Updated Content"}
    model.create_or_update_documents([new_doc], update=True)
    es_mock.search.assert_called_once_with(
        index=index_from_doc_id(new_doc["doc_id"]),
        body={
            "query": {"term": {"doc_id": {"value": 1}}}
        }
    )
    es_mock.delete.assert_called_once_with(index=index_from_doc_id(new_doc["doc_id"]), id="123")
    es_mock.index.assert_called_once()

def test_search():
    es_mock = MagicMock(spec=Elasticsearch)
    model = ElasticModel(es_mock)
    es_mock.search.return_value = {
        "hits": {"hits": [{"_id": "1", "_source": {"content": "value1"}}]}
    }
    embedded_search = [0.1, 0.2, 0.3]
    search_results = model.search(embedded_search, size=1)
    expected_output_from_es = {
        "size": 1,
        "query": {
            "script_score": {
                "query": {
                    "exists": {
                        "field": 'content_Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0_vectors'
                    }
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'content_Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0_vectors') + 1.0",
                    "params": {"query_vector": embedded_search}
                }
            }
        }
    }
    es_mock.search.assert_called_once_with(
        index=EMBEDDING_INDEX + "*", body=expected_output_from_es
    )
    assert search_results == [{"_id": "1", "_source": {"content": "value1"}}]


def test_search_hybrid_rrf_fusion_orders_by_combined_rank():
    es_mock = MagicMock(spec=Elasticsearch)
    model = ElasticModel(es_mock)
    # doc 1 is dense-only at rank 1; doc 2 is bm25-only at rank 1 -> both get the
    # same single best-rank contribution from their respective list, weighted 2:1.
    model.search = MagicMock(return_value=[{"_source": {"doc_id": 1}}])
    es_mock.search.return_value = {"hits": {"hits": [{"_source": {"doc_id": 2}}]}}

    results = model.search_hybrid([0.1, 0.2, 0.3], query_text="q")

    assert [hit["_source"]["doc_id"] for hit in results] == [1, 2]
    assert results[0]["_score"] == pytest.approx(2.0 / 61)
    assert results[1]["_score"] == pytest.approx(1.0 / 61)


def test_search_hybrid_dedupes_paragraph_level_hits_before_rrf():
    """
    A page indexed as multiple paragraphs can appear more than once in the same
    ranked list. RRF must take each document's single best rank per list, not sum
    a contribution per paragraph occurrence - otherwise a page with more indexed
    paragraphs wins purely from appearing more often, regardless of relevance.
    """
    es_mock = MagicMock(spec=Elasticsearch)
    model = ElasticModel(es_mock)
    # doc_id=2 is the single best dense hit (rank 1). doc_id=1 only reaches rank 2
    # and rank 3 (two different paragraphs of the same page). Correct RRF, using
    # doc_id=1's best rank (2) only, must still rank doc_id=2 first.
    model.search = MagicMock(return_value=[
        {"_source": {"doc_id": 2}},
        {"_source": {"doc_id": 1}},
        {"_source": {"doc_id": 1}},
    ])
    es_mock.search.return_value = {"hits": {"hits": []}}

    results = model.search_hybrid([0.1, 0.2, 0.3], query_text="q")

    assert [hit["_source"]["doc_id"] for hit in results] == [2, 1]
    assert results[1]["_score"] == pytest.approx(2.0 / 62)


def test_search_hybrid_doc_present_in_only_one_list():
    es_mock = MagicMock(spec=Elasticsearch)
    model = ElasticModel(es_mock)
    model.search = MagicMock(return_value=[{"_source": {"doc_id": 5}}])
    es_mock.search.return_value = {"hits": {"hits": []}}

    results = model.search_hybrid([0.1, 0.2, 0.3], query_text="q")

    assert [hit["_source"]["doc_id"] for hit in results] == [5]
    assert results[0]["_score"] == pytest.approx(2.0 / 61)
