#include "DataLoader.h"
#include "SimilarityComputer.h"
#include "PQ.h"
#include <iostream>

int main() {
    try {
        DataLoader loader;
        loader.load("test_data.csv");
        
        // 获取用户-商品交互矩阵
        auto& user_item_matrix = loader.get_user_item_matrix();
        
        // 训练PQ模型
        PQ pq_engine;
        std::vector<Eigen::VectorXf> item_vectors;
        for (const auto& vec : user_item_matrix) {
            item_vectors.emplace_back(Eigen::Map<const Eigen::VectorXf>(vec.data(), vec.size()));
        }
        pq_engine.train(item_vectors);
        
        // 相似度计算
        SimilarityComputer similarity;
        auto user_sim = similarity.user_similarity(
            Eigen::Map<const Eigen::MatrixXf>(user_item_matrix[0].data(), user_item_matrix.size(), user_item_matrix[0].size()));
        
        // 示例用户向量（实际应通过用户行为生成）
        Eigen::VectorXf query_vector(128);
        query_vector.setRandom();
        
        // 进行近似搜索
        auto results = pq_engine.search(query_vector, 5);
        
        // 输出结果
        std::cout << "找到最相似的前5个商品ID: ";
        for (auto id : results) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "运行时错误：" << e.what() << std::endl;
        return 1;
    }
    return 0;
}