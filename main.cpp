#include "DataLoader.h"
#include "SimilarityComputer.h"
#include "pq2/PQ.h"
#include <iostream>

int main() {
    try {
        // 原有数据加载模块
        DataLoader loader;
        loader.load("test_data.csv");

        // 新增量化器验证模块
        const int D = 128;  // 与当前数据维度一致
        const int M = 4;    // 子空间数量
        const int K = 256;  // 聚类数

        // 测试向量（从加载的数据中取样）
        std::vector<Eigen::VectorXf> test_vectors = {
            Eigen::VectorXf::Map(user_item_matrix[0].data(), D),
            Eigen::VectorXf::Map(user_item_matrix[1].data(), D),
            Eigen::VectorXf::Random(D)
        };

        // 初始化量化器
        PQ pq_verifier(D, M, K);
        pq_verifier.train(test_vectors);

        // 输出训练详情
        std::cout << "\n==== 量化器验证 ====" << std::endl;
        pq_verifier.print_cluster_centers_with_counts();
        pq_verifier.print_codebook();
        pq_verifier.print_data_point_assignments();
        
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
        
        // 验证查询功能
        auto [nearest_idx, min_dist] = pq_verifier.query(query_vector);
        std::cout << "\n==== 验证查询结果 ====" << std::endl;
        std::cout << "最近邻索引: " << nearest_idx << " 近似距离: " << min_dist << std::endl;

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
