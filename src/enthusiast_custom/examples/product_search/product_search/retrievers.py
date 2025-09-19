from django.db.models import QuerySet
from enthusiast_common.config.base import AgentConfig
from enthusiast_common.registry import BaseEmbeddingProviderRegistry
from enthusiast_common.retrievers import BaseVectorStoreRetriever
from enthusiast_common.structures import RepositoriesInstances
from langchain_core.language_models import BaseLanguageModel
from pgvector.django import CosineDistance


class ProductVectorStoreRetriever(BaseVectorStoreRetriever):
    def find_content_matching_query(self, query: str, keyword: str = "") -> QuerySet:
        embedding_vector = self._create_embedding_for_query(query)
        relevant_products = self._find_products_matching_vector(embedding_vector, keyword)
        return relevant_products

    def _create_embedding_for_query(self, query: str) -> list[float]:
        data_set = self.data_set_repo.get_by_id(self.data_set_id)
        embedding_provider = self.embeddings_registry.provider_for_dataset(self.data_set_id)
        return embedding_provider(data_set.embedding_model, data_set.embedding_vector_dimensions).generate_embeddings(
            query
        )

    def _find_products_matching_vector(
        self, embedding_vector: list[float], keyword: str
    ) -> QuerySet:
        embedding_distance = CosineDistance("embedding", embedding_vector)
        embeddings_with_products = self.model_chunk_repo.get_chunk_by_distance_and_keyword_for_data_set(
            self.data_set_id, embedding_distance, keyword
        )
        limited_embeddings_with_products = embeddings_with_products[: self.max_objects]
        return limited_embeddings_with_products

    @classmethod
    def create(
        cls,
        config: AgentConfig,
        data_set_id: int,
        repositories: RepositoriesInstances,
        embeddings_registry: BaseEmbeddingProviderRegistry,
        llm: BaseLanguageModel,
    ) -> BaseVectorStoreRetriever:
        return cls(
            data_set_id=data_set_id,
            data_set_repo=repositories.data_set,
            model_chunk_repo=repositories.product_chunk,
            embeddings_registry=embeddings_registry,
            **config.retrievers.product.extra_kwargs,
        )



class DocumentRetriever(BaseVectorStoreRetriever):
    def find_content_matching_query(self, query: str) -> QuerySet:
        embedding_vector = self._create_embedding_for_query(query)
        relevant_documents = self._find_documents_matching_vector(embedding_vector)
        return relevant_documents

    def _create_embedding_for_query(self, query: str) -> list[float]:
        data_set = self.data_set_repo.get_by_id(self.data_set_id)
        embedding_provider = self.embeddings_registry.provider_for_dataset(self.data_set_id)
        return embedding_provider(data_set.embedding_model, data_set.embedding_vector_dimensions).generate_embeddings(
            query
        )

    def _find_documents_matching_vector(self, embedding_vector: list[float]) -> QuerySet:
        embedding_distance = CosineDistance("embedding", embedding_vector)
        embeddings_with_documents = self.model_chunk_repo.get_chunk_by_distance_for_data_set(
            self.data_set_id, embedding_distance
        )
        limited_embeddings_with_documents = embeddings_with_documents[: self.max_objects]
        return limited_embeddings_with_documents

    @classmethod
    def create(
        cls,
        config: AgentConfig,
        data_set_id: int,
        repositories: RepositoriesInstances,
        embeddings_registry: BaseEmbeddingProviderRegistry,
        llm: BaseLanguageModel,
    ) -> BaseVectorStoreRetriever:
        return cls(
            data_set_id=data_set_id,
            data_set_repo=repositories.data_set,
            model_chunk_repo=repositories.document_chunk,
            embeddings_registry=embeddings_registry,
            **config.retrievers.document.extra_kwargs,
        )

