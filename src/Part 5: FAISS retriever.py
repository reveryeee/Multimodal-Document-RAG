class MultiDimRetriever:
    """
    The FAISS Retriever capable of handling multimodal and multi-dimensional vectors
    """

    def __init__(self, faiss_index_map):
        """
        faiss_index_map: dict
            key = Vector space
            value = {"index": faiss.Index, "meta": DataFrame}
        """
        self.faiss_index_map = faiss_index_map
        self.dim_list = sorted(faiss_index_map.keys())
        print(f"initialize MultiDimRetriever，available vector space size：{self.dim_list}")

    def search(self, query_vec, top_k=5):
        """
        query_vec: numpy array, shape=(dim,) 或 (1, dim)
        return: list of dict, include the meta info of the chunk + distance
        """
        query_vec = np.array(query_vec).flatten()
        dim = len(query_vec)

        if dim not in self.faiss_index_map:
            print(f"! did not find any corresponding index, dimension: {dim}")
            return []

        index = self.faiss_index_map[dim]["index"]
        meta_df = self.faiss_index_map[dim]["meta"]

        # keep shape=(1, dim)
        query_vec = query_vec.reshape(1, -1).astype('float32')

        # retrieve
        distances, indices = index.search(query_vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(meta_df):
                continue
            meta = meta_df.iloc[idx].to_dict()
            meta["distance"] = float(dist)
            results.append(meta)

        return results



# initialize MultiDimRetriever
retriever = MultiDimRetriever(faiss_index_map)


# The example below shows the retrieve with text embedding with vector space dimension 384

sample_query_idx = 0
sample_query_vec = embeddings_np[sample_query_idx] 

top_results = retriever.search(sample_query_vec, top_k=3)

print(f"\nTop 3 result：")
for i, res in enumerate(top_results):
    print(f"{i+1}. Page {res['page']} | Type: {res['chunk_type']} | distance: {res['distance']:.4f}")
    print(f"Content preview: {res['content_preview'][:200]}...\n")
