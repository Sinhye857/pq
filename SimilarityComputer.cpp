#include "SimilarityComputer.h"
#include <Eigen/Dense>

using namespace Eigen;

MatrixXf SimilarityComputer::user_similarity(const MatrixXf& user_matrix) {
    // 余弦相似度计算
    MatrixXf normalized = user_matrix.rowwise().normalized();
    return normalized * normalized.transpose();
}

MatrixXf SimilarityComputer::item_similarity(const MatrixXf& item_matrix) {
    // 商品相似度（转置后计算余弦）
    MatrixXf transposed = item_matrix.transpose();
    MatrixXf normalized = transposed.rowwise().normalized();
    return normalized * normalized.transpose();
}

MatrixXf SimilarityComputer::pearson_correlation(const MatrixXf& ratings_matrix) {
    MatrixXf centered = ratings_matrix;
    mean_centering(centered);
    return user_similarity(centered);
}

void SimilarityComputer::mean_centering(MatrixXf& matrix) {
    // 按行均值中心化
    VectorXf row_means = matrix.rowwise().mean();
    matrix.colwise() -= row_means;
}