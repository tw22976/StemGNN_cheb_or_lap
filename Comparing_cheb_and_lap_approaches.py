import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import csv



def cheb_polynomial(laplacian, max_order):
    """
    Compute the Chebyshev Polynomial, according to the graph laplacian.
    :param laplacian: the graph laplacian, [N, N].
    :return: the multi-order Chebyshev laplacian, [K, N, N].
    """
    N = laplacian.size(0)
    laplacian = laplacian.unsqueeze(0)
    first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
    laplacians = [first_laplacian, laplacian]

    for i in range(2, max_order + 1):
        if i % 2 == 0:
            new_laplacian = 2 * torch.matmul(laplacian, laplacians[i - 1]) - laplacians[i - 2]
        else:
            new_laplacian = 2 * torch.matmul(laplacian, laplacians[i - 1]) - laplacians[i - 2]
        laplacians.append(new_laplacian)

    multi_order_laplacian = torch.cat(laplacians, dim=0)
    return multi_order_laplacian



def lap_polynomial(laplacian, max_order):
    """
    Compute the polynomial approximation of the graph laplacian.
    :param laplacian: the graph laplacian, [N, N].
    :param max_order: the maximum order of the polynomial.
    :return: the multi-order polynomial approximation of the laplacian, [max_order, N, N].
    多项式近似，如基于多项式展开的谱方法。这种方法可以通过谱分解计算拉普拉斯矩阵的多项式函数，而无需逐个计算每个阶的多项式。
    """
    N = laplacian.size(0)
    polynomial_laplacians = [torch.eye(N, device=laplacian.device, dtype=laplacian.dtype),
                             laplacian]

    for k in range(2, max_order):
        term = 2 * torch.mm(laplacian, polynomial_laplacians[k - 1]) - polynomial_laplacians[k - 2]
        polynomial_laplacians.append(term)

    multi_order_laplacian = torch.stack(polynomial_laplacians, dim=0)
    return multi_order_laplacian


def calculate_metrics(original_matrix, approx_matrix):
    frobenius_norm = torch.norm(original_matrix - approx_matrix, 'fro').item()
    mse = ((original_matrix - approx_matrix) ** 2).mean().item()
    return frobenius_norm, mse














def save_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def main():
    N = 5  # Number of nodes
    num_repeats = 10
    max_order_range = range(11)  # Max order range: 0 to 10

    output_folder = 'output/Laplation Outcome'
    os.makedirs(output_folder, exist_ok=True)

    frobenius_data = []
    mse_data = []

    for repeat in range(1, num_repeats + 1):
        frobenius_row = []
        mse_row = []
        
        graph_structure = torch.rand(N, N)  # Random graph structure
        degree_matrix = torch.diag(torch.sum(graph_structure, dim=1))
        adjacency_matrix = graph_structure
        laplacian_matrix = degree_matrix - adjacency_matrix

        for max_order in max_order_range:
            mul_L_lap_polynomial = lap_polynomial(graph_structure, max_order)
            mul_L_cheb_polynomial = cheb_polynomial(graph_structure, max_order)
            # Calculate Frobenius norm and MSE 
            frobenius_lap_polynomial, mse_lap_polynomial = calculate_metrics(laplacian_matrix, mul_L_lap_polynomial)
            frobenius_cheb, mse_cheb = calculate_metrics(laplacian_matrix, mul_L_cheb_polynomial)
            # Append Frobenius norm and MSE values to rows
            frobenius_row.append(frobenius_lap_polynomial)
            mse_row.append(mse_lap_polynomial)
            # Append Frobenius norm and MSE values for Chebyshev polynomial
            frobenius_row.append(frobenius_cheb)
            mse_row.append(mse_cheb)

        frobenius_data.append(frobenius_row)
        mse_data.append(mse_row)

    # Save data to CSV files
    frobenius_filename = os.path.join(output_folder, 'Frobenius.csv')
    mse_filename = os.path.join(output_folder, 'MSE.csv')

    save_to_csv(frobenius_data, frobenius_filename)
    save_to_csv(mse_data, mse_filename)

if __name__ == '__main__':
    main()