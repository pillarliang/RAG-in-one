import os
from typing import List

from transformers import AutoModel
from numpy.linalg import norm
import numpy as np
import torch
import logging

from utility.tools import load_images_from_folder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JinaClipWrapper:
    def __init__(self, texts: List[str], images: List[str]):
        self.original_texts = texts
        self.original_images = images
        self.model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
        self.text_embeddings = self.model.encode_text(self.original_texts)
        self.image_embeddings = self.model.encode_image(self.original_images)
        self.weighted_lambda = 2

    def search_for_multimodal(self, query: str, top_k: int = 3):
        query_embedding = self.model.encode_text(query)
        text_similarity = self.cos_sim(query_embedding, self.text_embeddings)  # Example: [0.4649548  0.20749085 0.44108418]
        image_similarity = self.cos_sim(query_embedding, self.image_embeddings)  # Example: [-0.01970875  0.17084752]
        print(f"Text similarity scores:{text_similarity}")
        print(f"Image similarity scores:{image_similarity}")

        combined_similarity = np.concatenate((text_similarity, image_similarity * self.weighted_lambda))
        most_similar_indices = np.argsort(-combined_similarity)[:top_k]

        # store the most similar texts and images separately
        most_similar_texts = []
        most_similar_images = []

        # get the most similar texts and images based on the indices
        for index in most_similar_indices:
            if index < len(self.original_texts):
                most_similar_texts.append(self.original_texts[index])
            else:
                image_index = index - len(self.original_texts)
                most_similar_images.append(self.original_images[image_index])

        logger.info(f"Most similar texts:{most_similar_texts}")
        logger.info(f"Most similar images:{most_similar_images}")

        return {
            "texts": most_similar_texts,
            "images": most_similar_images
        }

    @staticmethod
    def cos_sim(a, b):
        """
        compute cosine similarity between two vectors
        :return:
        cosine similarity
        """
        dot_product = np.dot(b, a)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b, axis=1)

        return dot_product / (norm_a * norm_b)


if __name__ == "__main__":
    texts = ['A pig', 'A red cat', "A red pig"]
    image_folder = '../../model/image_dataset'
    images = load_images_from_folder(image_folder)

    jina_clip_wrapper = JinaClipWrapper(texts, images)
    query = "在 retrieval 之后应该做什么？"
    res = jina_clip_wrapper.search_for_multimodal(query)

    print(res)
