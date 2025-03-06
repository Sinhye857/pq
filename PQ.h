#pragma once
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>

class PQ {
public:
    PQ(size_t subspace_num = 8, size_t cluster_num = 256);

    // 训练PQ码本
    void train(const std::vector<Eigen::VectorXf>& item_vectors);

    // 向量编码
    std::vector<uint8_t> encode(const Eigen::VectorXf& vector) const;

    // 近似最近邻搜索
    std::vector<size_t> search(const Eigen::VectorXf& query, size_t topk) const;

private:
    void build_distance_table();

    size_t subspace_num_;     // 子空间数量
    size_t cluster_num_;      // 每子空间聚类数
    size_t sub_dim_ = 16;     // 子空间维度（128/8）

    std::vector<Eigen::MatrixXf> codebooks_;  // 码本存储
    std::unordered_map<size_t, Eigen::MatrixXf> distance_table_;  // 距离表
};