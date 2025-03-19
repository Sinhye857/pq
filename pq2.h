#pragma once
#include <vector>
#include <Eigen/Dense>

class PQ {
public:
    void train(const std::vector<Eigen::VectorXf>& item_vectors);
    std::vector<int> search(const Eigen::VectorXf& query_vector, int top_k);

private:
    // 添加必要的私有成员声明
    std::vector<Eigen::MatrixXf> codebooks_;
    std::vector<std::vector<uint8_t>> codes_;
};
