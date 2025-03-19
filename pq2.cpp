#include "pq2.h"
#include <iostream>

void PQ::train(const std::vector<Eigen::VectorXf>& item_vectors) {
    // 划分子空间并训练码本
    for (size_t i = 0; i < subspace_num_; ++i) {
        Eigen::MatrixXf subspace_vectors(item_vectors.size(), sub_dim_);
        for (size_t j = 0; j < item_vectors.size(); ++j) {
            subspace_vectors.row(j) = item_vectors[j].segment(i * sub_dim_, sub_dim_);
        }
        codebooks_.emplace_back(cluster_num_, sub_dim_);
        codebooks_.back().setRandom();
    }
    build_distance_table();
}

void PQ::print_cluster_centers_with_counts() {
    for (size_t i = 0; i < codebooks_.size(); ++i) {
        std::cout << "子空间" << i << "聚类中心:\n";
        std::cout << codebooks_[i] << "\n";
    }
}

void PQ::print_codebook() {
    std::cout << "编码簿内容:\n";
    for (const auto& codebook : codebooks_) {
        std::cout << codebook << "\n\n";
    }
}

void PQ::print_data_point_assignments() {
    std::cout << "数据点分配详情:\n";
    for (size_t i = 0; i < codes_.size(); ++i) {
        std::cout << "数据点" << i << ": ";
        for (auto code : codes_[i]) {
            std::cout << static_cast<int>(code) << " ";
        }
        std::cout << "\n";
    }
}

std::pair<size_t, float> PQ::query(const Eigen::VectorXf& query_vector) {
    size_t min_idx = 0;
    float min_dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < codes_.size(); ++i) {
        float dist = 0.0f;
        for (size_t j = 0; j < subspace_num_; ++j) {
            Eigen::VectorXf sub_query = query_vector.segment(j * sub_dim_, sub_dim_);
            dist += (codebooks_[j].row(codes_[i][j]) - sub_query).squaredNorm();
        }
        if (dist < min_dist) {
            min_dist = dist;
            min_idx = i;
        }
    }
    return {min_idx, min_dist};
}

std::vector<int> PQ::search(const Eigen::VectorXf& query_vector, int top_k) {
    std::vector<std::pair<float, int>> distances;
    for (size_t i = 0; i < codes_.size(); ++i) {
        float dist = 0.0f;
        for (size_t j = 0; j < subspace_num_; ++j) {
            Eigen::VectorXf sub_query = query_vector.segment(j * sub_dim_, sub_dim_);
            dist += (codebooks_[j].row(codes_[i][j]) - sub_query).squaredNorm();
        }
        distances.emplace_back(dist, i);
    }
    std::sort(distances.begin(), distances.end());
    std::vector<int> results;
    for (int i = 0; i < top_k && i < distances.size(); ++i) {
        results.push_back(distances[i].second);
    }
    return results;
}
