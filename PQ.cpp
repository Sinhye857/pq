#include "PQ.h"
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>

using namespace Eigen;

PQ::PQ(size_t subspace_num, size_t cluster_num) 
    : subspace_num_(subspace_num), cluster_num_(cluster_num),
      sub_dim_(128 / subspace_num) {}

void PQ::train(const std::vector<VectorXf>& item_vectors) {
    // 划分子空间并训练码本
    for (size_t i = 0; i < subspace_num_; ++i) {
        MatrixXf subspace_vectors(item_vectors.size(), sub_dim_);
        for (size_t j = 0; j < item_vectors.size(); ++j) {
            subspace_vectors.row(j) = item_vectors[j].segment(i * sub_dim_, sub_dim_);
        }
        
        // 执行K-means聚类（示例用随机初始化，实际应实现聚类算法）
        codebooks_.emplace_back(cluster_num_, sub_dim_);
        codebooks_.back().setRandom();
        
        // TODO: 实现实际聚类逻辑
    }
    build_distance_table();
}

std::vector<uint8_t> PQ::encode(const VectorXf& vector) const {
    std::vector<uint8_t> code(subspace_num_);
    for (size_t i = 0; i < subspace_num_; ++i) {
        VectorXf sub_vec = vector.segment(i * sub_dim_, sub_dim_);
        
        // 寻找最近聚类中心
        MatrixXf diff = codebooks_[i].rowwise() - sub_vec.transpose();
        Eigen::Index min_index;
        diff.rowwise().squaredNorm().minCoeff(&min_index);
        code[i] = static_cast<uint8_t>(min_index);
    }
    return code;
}

std::vector<size_t> PQ::search(const VectorXf& query, size_t topk) const {
    VectorXf dists = VectorXf::Zero(cluster_num_);
    
    // 计算各子空间距离
    for (size_t i = 0; i < subspace_num_; ++i) {
        VectorXf sub_query = query.segment(i * sub_dim_, sub_dim_);
        MatrixXf diff = codebooks_[i].rowwise() - sub_query.transpose();
        dists += diff.rowwise().squaredNorm();
    }
    
    // 获取topk索引
    std::vector<size_t> indices(dists.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + topk, indices.end(),
        [&dists](size_t a, size_t b) { return dists[a] < dists[b]; });
    
    return {indices.begin(), indices.begin() + topk};
}

void PQ::build_distance_table() {
    for (size_t i = 0; i < subspace_num_; ++i) {
        MatrixXf table(cluster_num_, cluster_num_);
        for (size_t j = 0; j < cluster_num_; ++j) {
            for (size_t k = 0; k < cluster_num_; ++k) {
                table(j, k) = (codebooks_[i].row(j) - codebooks_[i].row(k)).squaredNorm();
            }
        }
        distance_table_[i] = table;
    }
}