#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <set>
#include <string>
#include <numeric>


#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
//#include <CGAL/Simple_cartesian.h>
#include <CGAL/optimal_bounding_box.h>
#include<CGAL/Surface_mesh.h>
#include <CGAL/Bbox_3.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_ratio_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_profile.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Aff_transformation_3.h>

#include <boost/property_map/property_map.hpp>
#include <boost/graph/graph_traits.hpp>
#include <CGAL/linear_least_squares_fitting_3.h>
#include <CGAL/barycenter.h>
#include <CGAL/Aff_transformation_3.h>
#include <CGAL/Polygon_mesh_processing/transform.h>

#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_set_2.h>  
#include <CGAL/Barycentric_coordinates_2/Triangle_coordinates_2.h>

//#include <CGAL/Monge_via_jet_fitting.h> 
#include <CGAL/Polygon_mesh_processing/measure.h> 
#include <CGAL/boost/graph/iterator.h>



using namespace Eigen;
using namespace std;

constexpr double M_PI = 3.14159265358979323846;

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3						Point_3;
typedef CGAL::Surface_mesh<Point_3>			Surface_mesh;


//文件mompa_initialization.m

// mompa_initialization(SearchAgents_no, dim, ub, lb)
MatrixXd mompa_initialization(int SearchAgents_no, int dim,
	const VectorXd& ub, const VectorXd& lb) {
	if (ub.size() != lb.size()) {
		throw invalid_argument("Upper and lower bounds must have the same length.");
	}

	int Boundary_no = ub.size();
	MatrixXd Positions(SearchAgents_no, dim);

	random_device rd;
	mt19937 gen(42);
	uniform_real_distribution<double> dist(0.0, 1.0);

	// 情况1: 所有变量共享相同边界
	if (Boundary_no == 1) {
		double ub_val = ub(0);
		double lb_val = lb(0);
		for (int i = 0; i < SearchAgents_no; ++i) {
			for (int j = 0; j < dim; ++j) {
				Positions(i, j) = dist(gen) * (ub_val - lb_val) + lb_val;
			}
		}
	}

	// 情况2: 每个维度边界不同
	else if (Boundary_no > 1) {
		for (int j = 0; j < dim; ++j) {
			double ub_i = ub(j);
			double lb_i = lb(j);
			for (int i = 0; i < SearchAgents_no; ++i) {
				Positions(i, j) = dist(gen) * (ub_i - lb_i) + lb_i;
			}
		}
	}

	return Positions;
}


//文件UniformPoint.m

// 计算组合数 C(n,k)
long long nCr(int n, int k) {
	if (k > n) return 0;
	if (k == 0 || k == n) return 1;
	long long res = 1;
	for (int i = 1; i <= k; ++i) {
		res = res * (n - i + 1) / i;
	}
	return res;
}

// 生成 n 选 k 的组合（结果是所有组合的二维数组）
vector<vector<int>> combinations(int n, int k) {
	vector<vector<int>> result;
	vector<int> indices(k);
	for (int i = 0; i < k; ++i) indices[i] = i;

	while (true) {
		result.push_back(indices);
		int i;
		for (i = k - 1; i >= 0; --i) {
			if (indices[i] != i + n - k) break;
		}
		if (i < 0) break;
		++indices[i];
		for (int j = i + 1; j < k; ++j)
			indices[j] = indices[j - 1] + 1;
	}
	return result;
}

// UniformPoint(N, M)
pair<MatrixXd, int> UniformPoint(int N, int M) {
	int H1 = 1;
	while (nCr(H1 + M, M - 1) <= N)
		H1++;

	// === 生成第一层组合 ===
	auto combs1 = combinations(H1 + M - 1, M - 1);
	int rows1 = combs1.size();
	MatrixXd W(rows1, M);

	for (int i = 0; i < rows1; ++i) {
		for (int j = 0; j < M - 1; ++j)
			W(i, j) = combs1[i][j] - j - 1;
		W(i, M - 1) = H1;
	}

	// ([W, H1] - [0, W]) / H1
	for (int i = 0; i < rows1; ++i) {
		for (int j = M - 1; j >= 0; --j) {
			double prev = (j == 0 ? 0.0 : W(i, j - 1));
			W(i, j) = (W(i, j) - prev) / H1;
		}
	}

	// === 第二层（可选） ===
	if (H1 < M) {
		int H2 = 0;
		while (nCr(H1 + M - 1, M - 1) + nCr(H2 + M, M - 1) <= N)
			H2++;

		if (H2 > 0) {
			auto combs2 = combinations(H2 + M - 1, M - 1);
			int rows2 = combs2.size();
			MatrixXd W2(rows2, M);

			for (int i = 0; i < rows2; ++i) {
				for (int j = 0; j < M - 1; ++j)
					W2(i, j) = combs2[i][j] - j - 1;
				W2(i, M - 1) = H2;
			}

			for (int i = 0; i < rows2; ++i) {
				for (int j = M - 1; j >= 0; --j) {
					double prev = (j == 0 ? 0.0 : W2(i, j - 1));
					W2(i, j) = (W2(i, j) - prev) / H2;
				}
			}

			// 合并两层
			W.conservativeResize(rows1 + rows2, M);
			W.block(rows1, 0, rows2, M) = (W2.array() / 2.0 + 1.0 / (2.0 * M)).matrix();
		}
	}

	// 限制下界
	for (int i = 0; i < W.rows(); ++i)
		for (int j = 0; j < W.cols(); ++j)
			if (W(i, j) < 1e-6)
				W(i, j) = 1e-6;

	return { W, (int)W.rows() };
}




// 计算两个矩阵之间的欧式距离矩阵
MatrixXd pairwiseDistance(const MatrixXd& A, const MatrixXd& B) {
	int m = A.rows();
	int n = B.rows();
	MatrixXd D(m, n);

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			D(i, j) = (A.row(i) - B.row(j)).norm();
		}
	}
	return D;
}

//文件mompa_IGD.m

// 对应 MATLAB 函数 mompa_IGD(PopObj, PF)
double mompa_IGD(const MatrixXd& PopObj, const MatrixXd& PF) {
	MatrixXd D = pairwiseDistance(PF, PopObj);  // K×N
	VectorXd minDist(PF.rows());

	for (int i = 0; i < PF.rows(); ++i) {
		minDist(i) = D.row(i).minCoeff();
	}

	double Score = minDist.mean();  // 平均距离
	return Score;
}

//文件mompa_levy.m

//// 计算 gamma 函数（C++17 自带 tgamma）
//double gamma_func(double x) {
//	return tgamma(x);
//}

// 生成 Lévy 步长矩阵
MatrixXd mompa_levy(int n, int m, double beta) {
	// 计算 σ_u
	double num = tgamma(1 + beta) * sin(M_PI * beta / 2.0);
	double den = tgamma((1 + beta) / 2.0) * beta * pow(2.0, (beta - 1.0) / 2.0);

	double sigma_u = pow(num / den, 1.0 / beta);

	// 设置随机数生成器
	random_device rd;
	mt19937 gen(42);
	normal_distribution<double> normal_u(0.0, sigma_u);
	normal_distribution<double> normal_v(0.0, 1.0);

	MatrixXd z(n, m);

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			//double u = normal_u(gen);
			//double v = normal_v(gen);
			//z(i, j) = u / pow(fabs(v), 1.0 / beta);

			double u = normal_u(gen);
			double v = normal_v(gen);
			double v_abs = fabs(v);
			if (v_abs < 1e-12) v_abs = 1e-12;  // 防止除以0
			z(i, j) = u / pow(v_abs, 1.0 / beta);
		}
	}

	return z;
}


//文件mompa_Gaussian_search.m

// Gaussian search for MOMPA
MatrixXd mompa_Gaussian_search(MatrixXd Prey, int SearchAgents_no, int dim,
	const VectorXd& ub, const VectorXd& lb)
{
	std::random_device rd;
	std::mt19937 gen(42);
	std::normal_distribution<double> gaussian(0.0, 1.0);
	std::uniform_int_distribution<int> uniform_dim(0, dim - 1);

	// 1️⃣ 高斯扰动
	for (int i = 0; i < SearchAgents_no; ++i)
	{
		int d = uniform_dim(gen); // 随机选择一个维度
		double perturb = (ub(d) - lb(d)) * gaussian(gen);
		Prey(i, d) += perturb;
	}

	// 2️⃣ 边界检查
	for (int i = 0; i < Prey.rows(); ++i)
	{
		for (int j = 0; j < dim; ++j)
		{
			if (Prey(i, j) > ub(j))
				Prey(i, j) = ub(j);
			else if (Prey(i, j) < lb(j))
				Prey(i, j) = lb(j);
		}
	}

	return Prey;
}

//文件NDSort.m
 

std::pair<VectorXi, int> ENS_SS(const MatrixXd& PopObj, int nSort) {
	// ======= Step 1: unique rows =======
	std::vector<RowVectorXd> uniquePop;
	std::vector<int> Loc;
	std::unordered_map<std::string, int> uniqueMap;

	auto rowToKey = [&](const RowVectorXd& row) {
		std::string key;
		key.reserve(row.size() * 16);
		for (int j = 0; j < row.size(); ++j) {
			key += std::to_string(row(j)) + "|";
		}
		return key;
	};

	for (int i = 0; i < PopObj.rows(); ++i) {
		RowVectorXd row = PopObj.row(i);
		std::string key = rowToKey(row);
		if (uniqueMap.find(key) == uniqueMap.end()) {
			uniqueMap[key] = static_cast<int>(uniquePop.size());
			uniquePop.push_back(row);
		}
		Loc.push_back(uniqueMap[key]);
	}

	// ======= Step 2: emulate MATLAB unique(..., 'rows') sorting =======
	int nUnique = static_cast<int>(uniquePop.size());
	std::vector<int> order(nUnique);
	std::iota(order.begin(), order.end(), 0);

	std::sort(order.begin(), order.end(), [&](int a, int b) {
		for (int c = 0; c < PopObj.cols(); ++c) {
			if (uniquePop[a](c) < uniquePop[b](c)) return true;
			if (uniquePop[a](c) > uniquePop[b](c)) return false;
		}
		return false;
		});

	// 生成排序后的 uniquePopObj
	MatrixXd uniquePopObj(nUnique, PopObj.cols());
	for (int i = 0; i < nUnique; ++i)
		uniquePopObj.row(i) = uniquePop[order[i]];

	// 更新 Loc 到新的索引
	std::unordered_map<int, int> oldToNew;
	for (int i = 0; i < nUnique; ++i)
		oldToNew[order[i]] = i;
	for (int& l : Loc)
		l = oldToNew[l];

	// ======= Step 3: compute Table (histogram of Loc) =======
	VectorXi Table = VectorXi::Zero(nUnique);
	for (int loc : Loc) Table(loc)++;

	// ======= Step 4: initialize =======
	int N = uniquePopObj.rows();
	int M = uniquePopObj.cols();
	VectorXi FrontNo = VectorXi::Constant(N, std::numeric_limits<int>::max());
	int MaxFNo = 0;

	auto getSortedCount = [&]() {
		int count = 0;
		for (int i = 0; i < N; ++i)
			if (FrontNo(i) < std::numeric_limits<int>::max())
				count += Table(i);
		return count;
	};

	int totalIndividuals = static_cast<int>(Loc.size());
	int minCount = std::min(nSort, totalIndividuals);

	// ======= Step 5: main loop =======
	while (getSortedCount() < minCount) {
		MaxFNo = MaxFNo + 1;

		for (int i = 0; i < N; ++i) {
			if (FrontNo(i) == std::numeric_limits<int>::max()) {
				bool Dominated = false;

				for (int j = i - 1; j >= 0; --j) {
					if (FrontNo(j) == MaxFNo) {
						int m = 1; // 对应 MATLAB 的 m = 2
						while (m < M && uniquePopObj(i, m) >= uniquePopObj(j, m))
							m = m + 1;
						Dominated = (m >= M);
						if (Dominated || M == 2)
							break;
					}
				}

				if (!Dominated)
					FrontNo(i) = MaxFNo;
			}
		}
	}

	// ======= Step 6: map back to original individuals =======
	VectorXi OriginalFrontNo(Loc.size());
	for (int i = 0; i < Loc.size(); ++i)
		OriginalFrontNo(i) = FrontNo(Loc[i]);

	return std::make_pair(OriginalFrontNo, MaxFNo);
}

std::pair<Eigen::VectorXi, int> ENS_SS4(const Eigen::MatrixXd& PopObj, int nSort) {
	std::vector<Eigen::VectorXd> uniquePop;
	std::vector<int> Loc;
	std::unordered_map<std::string, int> uniqueMap;

	for (int i = 0; i < PopObj.rows(); ++i) {
		std::string key;
		for (int j = 0; j < PopObj.cols(); ++j) {
			key += std::to_string(PopObj(i, j)) + "|";
		}

		if (uniqueMap.find(key) == uniqueMap.end()) {
			uniqueMap[key] = uniquePop.size();
			uniquePop.push_back(PopObj.row(i));
		}
		Loc.push_back(uniqueMap[key]);
	}

	// 创建去重后的 PopObj 矩阵
	Eigen::MatrixXd uniquePopObj(uniquePop.size(), PopObj.cols());
	for (int i = 0; i < uniquePop.size(); ++i) {
		uniquePopObj.row(i) = uniquePop[i];
	}

	// 计算 Table（出现次数）
	Eigen::VectorXi Table = Eigen::VectorXi::Zero(uniquePop.size());
	for (int loc : Loc) {
		Table(loc)++;
	}

	// 获取去重后尺寸
	int N = uniquePopObj.rows();
	int M = uniquePopObj.cols();
	// 初始化前沿编号 - 改为 VectorXi
	Eigen::VectorXi FrontNo = Eigen::VectorXi::Constant(N, std::numeric_limits<int>::max());
	// 初始化最大前沿编号
	int MaxFNo = 0;

	// 计算已排序个体数量
	auto getSortedCount = [&]() {
		int count = 0;
		for (int i = 0; i < N; ++i) {
			if (FrontNo(i) < std::numeric_limits<int>::max()) {
				count += Table(i);
			}
		}
		return count;
	};

	int totalIndividuals = Loc.size();  // length(Loc)
	int minCount = std::min(nSort, totalIndividuals);

	while (getSortedCount() < minCount) {
		MaxFNo = MaxFNo + 1;

		for (int i = 0; i < N; ++i) {
			if (FrontNo(i) == std::numeric_limits<int>::max()) {
				bool Dominated = false;

				// 从i-1向前遍历到0
				for (int j = i - 1; j >= 0; --j) {
					if (FrontNo(j) == MaxFNo) {
						int m = 1;  // C++索引从0开始，这里从第2个目标开始（索引1）
						while (m < M && uniquePopObj(i, m) >= uniquePopObj(j, m)) {
							m = m + 1;
						}
						Dominated = (m >= M);  // 如果检查完所有目标，说明被支配

						if (Dominated || M == 2) {
							break;
						}
					}
				}

				if (!Dominated) {
					FrontNo(i) = MaxFNo;
				}
			}
		}
	}

	// 将去重后的前沿编号映射回原始种群 - 改为 VectorXi
	Eigen::VectorXi OriginalFrontNo(Loc.size());
	for (int i = 0; i < Loc.size(); ++i) {
		OriginalFrontNo(i) = FrontNo(Loc[i]);
	}

	return std::make_pair(OriginalFrontNo, MaxFNo);
}

std::pair<VectorXi, int> NDSort(MatrixXd PopObj, const MatrixXd& PopCon, int nSort) {
	int N = PopObj.rows();
	int M = PopObj.cols();

	VectorXd Infeasible = (PopCon.array() > 0).rowwise().any().cast<double>();

	// 计算每个目标函数的当前最大值
	VectorXd maxPopObj = PopObj.colwise().maxCoeff();

	// 计算不可行解的约束违反总值
	VectorXd constraintSum = (PopCon.array().max(0)).rowwise().sum();

	// 处理不可行解：目标值 = 当前最大值 + 约束违反值
	for (int i = 0; i < N; ++i) {
		if (Infeasible(i) > 0) {
			PopObj.row(i) = maxPopObj.transpose() + RowVectorXd::Constant(1, M, constraintSum(i));
		}
	}

	if (M < 3 || N < 500) {
		return ENS_SS(PopObj, nSort);
	}
	else {
		// TODO: T-ENS 树型排序，可替换 ENS_SS
		return ENS_SS(PopObj, nSort);
	}
}
 

// ---------------- NDSort ----------------
pair<VectorXi, int> NDSort(const MatrixXd& PopObj, int nSort = numeric_limits<int>::max()) {
	int N = PopObj.rows();
	int M = PopObj.cols();

	if (M < 3 || N < 500) {
		return ENS_SS(PopObj, nSort);
	}
	else {
		// TODO: T-ENS 树型排序，可替换 ENS_SS
		return ENS_SS(PopObj, nSort);
	}
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> DTLZ4_2(int numObj, Eigen::MatrixXd decisionVar) {
	int M = numObj; 
	int N = decisionVar.rows();
	int D = decisionVar.cols();
	 
	// 对前 M-1 列进行100次幂运算
	decisionVar.leftCols(M - 1) = decisionVar.leftCols(M - 1).array().pow(100);

	// 计算 g 值：对第M列到最后一列，每个元素减0.5后平方，然后按行求和
	Eigen::VectorXd g=(decisionVar.rightCols(D - M).array() - 0.5).square().rowwise().sum();

	const double pi_half = M_PI / 2.0;

	Eigen::MatrixXd fobj(N, M);


	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < M; ++j) {
			// 对应MATLAB中的列索引：fliplr改变了列顺序
			int matlab_j = M - 1 - j;  // 因为fliplr

			double product = 1.0;

			// 累积乘积部分（考虑fliplr的影响）
			for (int k = 0; k < matlab_j; ++k) {
				if (k < decisionVar.cols()) {
					product *= std::cos(decisionVar(i, k) * pi_half);
				}
			}

			// 正弦部分
			if (matlab_j < M - 1) {
				int sin_index = M - 2 - matlab_j;  // 因为M-1:-1:1的反向索引
				if (sin_index < decisionVar.cols()) {
					product *= std::sin(decisionVar(i, sin_index) * pi_half);
				}
			}

			fobj(i, j) = (1.0 + g(i)) * product;
		}
	}



	MatrixXd cosPart = MatrixXd::Ones(N, M);
	for (int j = 1; j < M; ++j) {
		int col = j - 1;
		if (col < D)
			cosPart.col(j) = (decisionVar.col(col).array() * pi_half).cos();
	}

	// cumprod(...,2)
	for (int j = 1; j < M; ++j)
		cosPart.col(j) = cosPart.col(j - 1).array() * cosPart.col(j).array();

	// fliplr(...)
	MatrixXd flippedCumprod = cosPart.rowwise().reverse();

	// [ones(N,1), sin(x(:,M-1:-1:1)*pi/2)]
	MatrixXd sinPart = MatrixXd::Ones(N, M);
	for (int j = 1; j < M; ++j) {
		int srcCol = M - j - 1;  // 对应 MATLAB 的 M-1:-1:1
		if (srcCol < D && srcCol >= 0)
			sinPart.col(j) = (decisionVar.col(srcCol).array() * pi_half).sin();
	}

	// fobj = repmat(1+g,1,M) .* flippedCumprod .* sinPart
	MatrixXd fobj = (g.replicate(1, M).array() + 1.0)
		* flippedCumprod.array() * sinPart.array();


	// 创建约束矩阵，大小为 numRows × (2*D)
	Eigen::MatrixXd fcon(N, 2 * D);

	// 第一组约束：fcon(:, 1:D) = -decisionVar;
	fcon.leftCols(D) = -decisionVar;

	// 第二组约束：fcon(:, D+1: 2*D) = decisionVar - 1;
	fcon.rightCols(D) = decisionVar - Eigen::MatrixXd::Ones(N, D);
	//fcon.rightCols(D) = decisionVar.array() - 1.0;

	auto [P,_] = UniformPoint(N, M);

	// P = P./repmat(sqrt(sum(P.^2,2)),1,M);
	Eigen::VectorXd norms = P.rowwise().norm();  // 计算每行的2-范数 (sqrt(sum(P.^2,2)))
	P = P.array().colwise() / norms.array();     // 每行除以其范数

	return std::make_tuple(fobj, fcon, P);

}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> DTLZ4(const Eigen::MatrixXd& decisionVar, int numObj ) {
	int D = decisionVar.cols();
	int N = decisionVar.rows();
	int M = numObj;
	int populationSize = decisionVar.rows();

	// 检查输入有效性
	if (populationSize == 0 || D == 0 || M == 0) {
		throw std::invalid_argument("Empty input matrix");
	}

	if (M > D) {
		throw std::invalid_argument("numObj cannot be greater than decision variable dimensions");
	}

	Eigen::MatrixXd modifiedVar = decisionVar;

	// 对前 M-1 维进行幂运算
	if (M > 1) {
		modifiedVar.block(0, 0, populationSize, M - 1) =
			modifiedVar.block(0, 0, populationSize, M - 1).array().pow(100);
	}

	// 计算 g 函数
	Eigen::VectorXd g = Eigen::VectorXd::Zero(populationSize);
	if (D - M + 1 > 0) {
		g = (modifiedVar.block(0, M - 1, populationSize, D - M + 1).array() - 0.5)
			.square()
			.rowwise()
			.sum();
	}

	// 初始化目标函数矩阵
	Eigen::MatrixXd fobj(populationSize, M);

	if (M == 1) {
		// 单目标特殊情况
		fobj.col(0) = 1.0 + g.array();
	}
	else {
		// 多目标情况
		Eigen::MatrixXd angles = modifiedVar.block(0, 0, populationSize, M - 1) * (M_PI / 2.0);

		for (int i = 0; i < populationSize; ++i) {
			for (int j = 0; j < M; ++j) {
				double prod = 1.0;
				// 累乘前 (M - j - 1) 个 cos
				for (int k = 0; k < M - j - 1; ++k) {
					prod *= std::cos(angles(i, k));
				}
				// 乘上 sin(angles(i, M - j - 1))，除非是第一个目标
				double sin_term = (j == 0) ? 1.0 : std::sin(angles(i, M - j - 1));
				fobj(i, j) = (1.0 + g(i)) * prod * sin_term;
			}
		}

	}

	// 约束处理
	Eigen::MatrixXd fcon(populationSize, 2 * D);
	fcon.leftCols(D) = -decisionVar;
	fcon.rightCols(D) = decisionVar.array() - 1.0;

	// Pareto 前沿 - 添加安全检查
	Eigen::MatrixXd P;
	try {
		auto [P,_] = UniformPoint(N, M);

		// 检查 P 是否为空或包含无效值
		if (P.rows() == 0 || P.cols() == 0) {
			// 生成默认的 Pareto 前沿点
			P = Eigen::MatrixXd::Ones(N, M);
			for (int i = 0; i < M; ++i) {
				P.col(i) = P.col(i) * (i + 1) / M;
			}
		}

		// 计算范数前检查矩阵是否有效
		Eigen::VectorXd norm = P.rowwise().norm();

		// 检查并处理零范数
		for (int i = 0; i < norm.size(); ++i) {
			if (norm(i) == 0 || !std::isfinite(norm(i))) {
				norm(i) = 1.0;
			}
		}

		// 归一化
		for (int i = 0; i < M; ++i) {
			P.col(i) = P.col(i).array() / norm.array();
		}

	}
	catch (const std::exception& e) {
		// 如果 UniformPoint 失败，使用默认值
		P = Eigen::MatrixXd::Ones(N, M);
		for (int i = 0; i < M; ++i) {
			P.col(i) = P.col(i) * (i + 1) / M;
		}
	}

	return std::make_tuple(fobj, fcon,P);
}


tuple<Vector2d, double, MatrixXd> rectangleLayoutProblem(
	const VectorXd& X,
	const VectorXd& W,
	const VectorXd& H,
	const MatrixXd& connectivity = MatrixXd()) {

	int N = W.size(); // 矩形块数量

	// 验证输入维度
	if (X.size() != 2 * N) {
		throw invalid_argument("决策向量X的维度应为2N，其中N为矩形块数量");
	}
	if (W.size() != H.size()) {
		throw invalid_argument("宽度向量W和高度向量H的维度必须相同");
	}

	// 提取坐标
	vector<double> x(N), y(N);
	for (int i = 0; i < N; ++i) {
		x[i] = X(2 * i);
		y[i] = X(2 * i + 1);
	}

	// ==================== 目标函数1: 最小化包围矩形面积 ====================
	double max_x_plus_w = 0.0;
	double max_y_plus_h = 0.0;

	for (int i = 0; i < N; ++i) {
		max_x_plus_w = max(max_x_plus_w, x[i] + W(i));
		max_y_plus_h = max(max_y_plus_h, y[i] + H(i));
	}

	double f1 = max_x_plus_w * max_y_plus_h;

	// ==================== 目标函数2: 最小化通道总长度 ====================
	double f2 = 0.0;

	// 如果提供了连接矩阵，计算通道长度
	if (connectivity.size() > 0) {
		if (connectivity.rows() != N || connectivity.cols() != N) {
			throw invalid_argument("连接矩阵必须是N×N的方阵");
		}

		for (int i = 0; i < N - 1; ++i) {
			for (int j = i + 1; j < N; ++j) {
				if (connectivity(i, j) > 0) { // δ_ij = 1表示有连接
					// 计算通道长度L_ij (使用中心点之间的欧几里得距离作为简化)
					double center_x_i = x[i] + W(i) / 2.0;
					double center_y_i = y[i] + H(i) / 2.0;
					double center_x_j = x[j] + W(j) / 2.0;
					double center_y_j = y[j] + H(j) / 2.0;

					double L_ij = sqrt(pow(center_x_i - center_x_j, 2) +
						pow(center_y_i - center_y_j, 2));

					f2 += connectivity(i, j) * L_ij;
				}
			}
		}
	}

	// ==================== 约束条件: 矩形块不重叠 ====================
	double constraint_violation = 0.0;

	for (int i = 0; i < N - 1; ++i) {
		for (int j = i + 1; j < N; ++j) {
			// 计算x方向重叠
			double x_overlap = max(0.0, min(x[i] + W(i), x[j] + W(j)) - max(x[i], x[j]));
			// 计算y方向重叠
			double y_overlap = max(0.0, min(y[i] + H(i), y[j] + H(j)) - max(y[i], y[j]));
			// 重叠面积
			double overlap_area = x_overlap * y_overlap;

			constraint_violation += overlap_area;
		}
	}

	// 非负约束: x_i ≥ 0, y_i ≥ 0
	for (int i = 0; i < N; ++i) {
		if (x[i] < 0) constraint_violation += -x[i] * 10.0; // 惩罚项
		if (y[i] < 0) constraint_violation += -y[i] * 10.0; // 惩罚项
	}

	// ==================== 返回结果 ====================
	Vector2d objectives;
	objectives << f1, f2;

	// 雅可比矩阵（可选，用于梯度计算）
	MatrixXd jacobian = MatrixXd::Zero(2, 2 * N); // 简化处理

	return make_tuple(objectives, constraint_violation, jacobian);
}

MatrixXd generateReferencePointsForRectangleLayout(int N) {
	// 简化的参考点生成
	// 在实际应用中，这应该基于问题特性生成

	int numPoints = 100;
	MatrixXd P(numPoints, 2);

	// 生成在目标空间均匀分布的点
	for (int i = 0; i < numPoints; ++i) {
		double t = static_cast<double>(i) / (numPoints - 1);
		// 第一个目标（面积）的范围估计
		double f1_min = N * 1.0; // 最小可能面积（假设每个矩形面积为1）
		double f1_max = N * 100.0; // 最大可能面积估计

		// 第二个目标（通道长度）的范围估计
		double f2_min = 0.0;
		double f2_max = N * (N - 1) / 2 * sqrt(2.0) * 10.0; // 最大通道长度估计

		P(i, 0) = f1_min + t * (f1_max - f1_min);
		P(i, 1) = f2_min + (1 - t) * (f2_max - f2_min); // 与f1权衡
	}

	return P;
}

tuple<MatrixXd, MatrixXd, MatrixXd> rectangleLayoutMOFcn(
	const MatrixXd& decisionVar,
	const VectorXd &W,
	const VectorXd &H,
	const MatrixXd& connectivity) {

	int N = decisionVar.rows(); // 解的数量 

	if (W.size() == 0 || H.size() == 0) {
		throw invalid_argument("必须提供宽度向量W和高度向量H");
	}

	int numRectangles = W.size();
	int numObjectives = 2;

	MatrixXd fobj(N, numObjectives);
	MatrixXd fcon(N, 1); // 单约束（总违反度）
	MatrixXd P; // 参考点（可选）

	for (int i = 0; i < N; ++i) {
		VectorXd X = decisionVar.row(i);
		auto [objectives, constraint_violation, jacobian] = rectangleLayoutProblem(X, W, H, connectivity);

		fobj.row(i) = objectives;
		fcon(i, 0) = constraint_violation;
	}

	// 生成参考点（Pareto前沿的近似）
	P = generateReferencePointsForRectangleLayout(numRectangles);

	return make_tuple(fobj, fcon, P);
}

 

tuple<MatrixXd, MatrixXd, MatrixXd> mompa_getMOFcn(const string& F,
	const MatrixXd& decisionVar,
	int numObj) {
	int N = decisionVar.rows();
	int D = decisionVar.cols();

	MatrixXd fobj, fcon, P;

	if (F == "ZDT3") {
		fobj = MatrixXd(N, 2);
		fobj.col(0) = decisionVar.col(0);

		MatrixXd g = MatrixXd::Ones(N, 1) + 9 * decisionVar.rightCols(D - 1).rowwise().mean();
		MatrixXd h = 1 - (fobj.col(0).array() / g.array()).sqrt()
			- (fobj.col(0).array() / g.array()) * sin(10 * M_PI * fobj.col(0).array());
		fobj.col(1) = g.array() * h.array();

		// 约束处理
		fcon = MatrixXd(N, 2 * D);
		fcon.leftCols(D) = -decisionVar;
		fcon.rightCols(D) = decisionVar - MatrixXd::Ones(N, D);

		// Pareto前沿
		P = MatrixXd(N, 2);
		VectorXd x = VectorXd::LinSpaced(N, 0, 1);
		P.col(0) = x;
		P.col(1) = 1 - x.array().sqrt() - x.array() * sin(10 * M_PI * x.array());

		// 非支配排序筛选
		pair<VectorXi, int> sortResult = NDSort(P,fcon, 1);
		VectorXi fronts = sortResult.first;

		// 正确创建布尔掩码
		VectorXd::Index front1Count = 0;
		for (int i = 0; i < fronts.size(); ++i) {
			if (fronts(i) == 1) {
				front1Count++;
			}
		}

		MatrixXd P_front1(front1Count, 2);
		int idx = 0;
		for (int i = 0; i < fronts.size(); ++i) {
			if (fronts(i) == 1) {
				P_front1.row(idx++) = P.row(i);
			}
		}

		P = P_front1;
	}
	else if (F == "DTLZ4") {
		std::tie(fobj, fcon, P) = DTLZ4_2(numObj, decisionVar);
		 
	}

	return make_tuple(fobj, fcon, P);
}


//文件EnvironmentalSelection.m

VectorXi LastSelection(const MatrixXd& PopObj1, const MatrixXd& PopObj2,
	int K, const MatrixXd& Z, const RowVectorXd& Zmin)
{
	MatrixXd PopObj(PopObj1.rows() + PopObj2.rows(), PopObj1.cols());
	PopObj << PopObj1, PopObj2;
	PopObj.rowwise() -= Zmin;

	int N = PopObj.rows();
	int M = PopObj.cols();
	int N1 = PopObj1.rows();
	int N2 = PopObj2.rows();
	int NZ = Z.rows();

	// ---- 1. 归一化 ----
	MatrixXd w = MatrixXd::Identity(M, M).array() + 1e-6;
	VectorXi Extreme(M);
	for (int i = 0; i < M; ++i)
	{
		MatrixXd div = PopObj.array().rowwise() / w.row(i).array();
		VectorXd maxVals = div.rowwise().maxCoeff();
		int idx;
		maxVals.minCoeff(&idx);
		Extreme(i) = idx;
	}

	// 超平面截距
	//VectorXd a;
	//if (PopObj.rows() >= M) {
	//	// 检查索引是否有效
	//	if (Extreme(0) >= 0 && Extreme(0) + M <= PopObj.rows()) {
	//		MatrixXd block = PopObj.block(Extreme(0), 0, M, M);
	//		// 检查矩阵是否可逆
	//		if (block.determinant() != 0) {
	//			a = block.colPivHouseholderQr().solve(VectorXd::Ones(M));
	//		}
	//		else {
	//			a = VectorXd::Ones(M);  // 退化解
	//		}
	//	}
	//	else {
	//		a = VectorXd::Ones(M);  // 索引越界，使用退化解
	//	}
	//}
	//else {
	//	a = VectorXd::Ones(M);
	//}

	// 计算超平面截距
	MatrixXd block(M, M);
	for (int i = 0; i < M; ++i)
		block.row(i) = PopObj.row(Extreme(i));

	VectorXd a;
	if (block.determinant() != 0)
		a = block.colPivHouseholderQr().solve(VectorXd::Ones(M));
	else
		a = VectorXd::Ones(M);

	a = a.cwiseInverse();
	for (int i = 0; i < a.size(); ++i)
		if (std::isnan(a(i))) a(i) = PopObj.col(i).maxCoeff();

	for (int i = 0; i < M; ++i)
		PopObj.col(i).array() /= a(i);

	// ---- 2. 计算与参考向量的余弦相似度 ----
	//MatrixXd Cosine = MatrixXd::Zero(N, NZ);
	//for (int i = 0; i < N; ++i)
	//	for (int j = 0; j < NZ; ++j)
	//	{
	//		double dotp = PopObj.row(i).dot(Z.row(j));
	//		double denom = PopObj.row(i).norm() * Z.row(j).norm() + 1e-12;
	//		Cosine(i, j) = 1.0 - (dotp / denom);
	//	}

	//MatrixXd Distance = MatrixXd::Zero(N, NZ);
	//for (int i = 0; i < N; ++i)
	//	for (int j = 0; j < NZ; ++j)
	//	{
	//		double normP = PopObj.row(i).norm();
	//		Distance(i, j) = normP * sqrt(1.0 - pow(1.0 - Cosine(i, j), 2));
	//	}

	MatrixXd Cosine = MatrixXd::Zero(N, NZ);
	MatrixXd Distance = MatrixXd::Zero(N, NZ);
	for (int i = 0; i < N; ++i) {
		double normP = PopObj.row(i).norm() + 1e-12;
		for (int j = 0; j < NZ; ++j) {
			double dotp = PopObj.row(i).dot(Z.row(j));
			double normZ = Z.row(j).norm() + 1e-12;
			double cosine = dotp / (normP * normZ);
			Cosine(i, j) = cosine;
			Distance(i, j) = normP * sqrt(1.0 - cosine * cosine);
		}
	}

	// 每个个体对应最近的参考向量
	VectorXi pi(N);
	VectorXd d(N);
	for (int i = 0; i < N; ++i)
	{
		int idx;
		d(i) = Distance.row(i).minCoeff(&idx);
		pi(i) = idx;
	}

	// ---- 3. 计算每个参考点已关联解的数量 ----
	VectorXi rho = VectorXi::Zero(NZ);
	//for (int i = 0; i < N1; ++i)
	//	rho(pi(i))++;
	for (int i = 0; i < N1; ++i)
		if (pi(i) >= 0 && pi(i) < NZ)
			rho(pi(i))++;

	// ---- 4. 环境选择 ----
	std::random_device rd;
	std::mt19937 gen(42);
	VectorXi Choose = VectorXi::Zero(N2);
	vector<bool> Zchoose(NZ, true);

	int chosenCount = 0;
	while (chosenCount < K)
	{
		vector<int> Temp;
		for (int i = 0; i < NZ; ++i)
			if (Zchoose[i])
				Temp.push_back(i);

		// 找到最少关联点的参考向量
		int minRho = INT_MAX;
		for (int j : Temp)
			minRho = min(minRho, rho(j));

		vector<int> Jmin;
		for (int j : Temp)
			if (rho(j) == minRho)
				Jmin.push_back(j);

		//int j = Jmin[rand() % Jmin.size()];
		std::uniform_int_distribution<> dist_j(0, Jmin.size() - 1);
		int j = Jmin[dist_j(gen)];

		// 找到该参考点关联的最后层解
		vector<int> I;
		for (int i = 0; i < N2; ++i)
			if (Choose(i) == 0 && pi(N1 + i) == j)
				I.push_back(i);

		if (!I.empty())
		{
			/*int s = (rho(j) == 0)
				? int(min_element(I.begin(), I.end(), [&](int a, int b) { return d(N1 + a) < d(N1 + b); }) - I.begin())
				: rand() % I.size();*/
			int s;
			if (rho(j) == 0)
			{
				// 找到最小距离的索引
				auto min_it = min_element(I.begin(), I.end(),
					[&](int a, int b) { return d(N1 + a) < d(N1 + b); });
				s = distance(I.begin(), min_it);
			}
			else
			{
				//s = rand() % I.size();
				std::uniform_int_distribution<> dist_s(0, I.size() - 1);
				s = dist_s(gen);
			}
			Choose(I[s]) = 1;
			rho(j)++;
			chosenCount++;
		}
		else
		{
			Zchoose[j] = false;
		}
	}

	return Choose;
}

// ---------- 工具函数：根据索引选取行 ----------
MatrixXd selectRows(const MatrixXd& A, const vector<int>& idxs) {
	MatrixXd out(idxs.size(), A.cols());
	for (int i = 0; i < idxs.size(); ++i)
		out.row(i) = A.row(idxs[i]);
	return out;
}

MatrixXd EnvironmentalSelection(
	const std::string& FUN,
	const MatrixXd& Population,
	int N, int M,
	const MatrixXd& Z,
	const RowVectorXd& Zmin)
{
	RowVectorXd Zmin_use;
	if (Zmin.size() == 0) {
		Zmin_use = RowVectorXd::Ones(Z.cols());
	}
	else {
		Zmin_use = Zmin;
	}

	// 计算目标函数值
	auto [Population_objs, _, __] = mompa_getMOFcn(FUN, Population, M); 

	// 非支配排序
	auto [FrontNo, MaxFNo] = NDSort(Population_objs, N);

		// 找出属于下一代的个体
	vector<int> idxNext, idxLast;
	for (int i = 0; i < FrontNo.size(); ++i) {
		if (FrontNo(i) < MaxFNo)
			idxNext.push_back(i);
		else if (FrontNo(i) == MaxFNo)
			idxLast.push_back(i);
	}

	int K = N - static_cast<int>(idxNext.size());
	MatrixXd PopObj1 = selectRows(Population_objs, idxNext);
	MatrixXd PopObj2 = selectRows(Population_objs, idxLast);



	VectorXi Choose = LastSelection(PopObj1, PopObj2, K, Z, Zmin_use);

	// 构造下一代个体索引
	vector<int> NextIdx;
	// 添加前 MaxFNo-1 层的个体
	for (int i = 0; i < FrontNo.size(); ++i) {
		if (FrontNo(i) < MaxFNo) {
			NextIdx.push_back(i);
		}
	}
	// 添加从最后一层选择的个体
	//for (int i = 0; i < Choose.size(); ++i) {
	for (int i = 0; i < idxLast.size(); ++i){
		if (Choose(i)) {
			//NextIdx.push_back(Last[i]);
			NextIdx.push_back(idxLast[i]);
		}
	}

	// 提取下一代种群
	MatrixXd NextPop(NextIdx.size(), Population.cols());
	for (int i = 0; i < NextIdx.size(); ++i) {
		NextPop.row(i) = Population.row(NextIdx[i]);
	}

	return NextPop;
}




// MOMPA 主函数


// Helper: convert a row to a string key
string row_to_string(const RowVectorXd& row) {
	stringstream ss;
	for (int i = 0; i < row.size(); ++i) {
		ss << row(i) << ",";
	}
	return ss.str();
}

// unique_rows function
MatrixXd unique_rows(const MatrixXd& mat) {
	set<string> seen;
	vector<RowVectorXd> unique_vec;
	for (int i = 0; i < mat.rows(); ++i) {
		string key = row_to_string(mat.row(i));
		if (seen.find(key) == seen.end()) {
			unique_vec.push_back(mat.row(i));
			seen.insert(key);
		}
	}
	MatrixXd result(unique_vec.size(), mat.cols());
	for (int i = 0; i < unique_vec.size(); ++i) {
		result.row(i) = unique_vec[i];
	}
	return result;
}
 

tuple<MatrixXd, vector<double>, MatrixXd> mompa ( 
	string FUN,
	int SearchAgents_no,
	int dim,
	const RowVectorXd& lb,
	const RowVectorXd& ub,
	int numObj,
	int Max_iter)
{
	std::random_device rd;
	std::mt19937 gen(42);
	std::uniform_real_distribution<> dis(0.0, 1.0);
	std::uniform_int_distribution<int> int_dis(0, SearchAgents_no - 1);  
 

	int Iter = 0;
	double FADs = 0.2;
	double P = 0.5;
	vector<double> sub_IGD(Max_iter, 0.0);
	MatrixXd fit = MatrixXd::Constant(SearchAgents_no, numObj, numeric_limits<double>::infinity());
	MatrixXd P_1;

	MatrixXd Prey, Z, subfit, Zmin, Prey_old;
	while (Iter < Max_iter) {
		if (Iter == 0) {
			Prey= mompa_initialization(SearchAgents_no, dim, ub, lb);
			auto temp = UniformPoint(SearchAgents_no, numObj);
			Z = temp.first;
			SearchAgents_no = temp.second; 
			std::tie(subfit, std::ignore, std::ignore) = mompa_getMOFcn(FUN, Prey, numObj);
			Zmin = subfit.colwise().minCoeff();
			Prey=EnvironmentalSelection(FUN, Prey, SearchAgents_no, numObj, Z, Zmin);
			Prey_old = Prey;
		}
		MatrixXd Xmin = lb.replicate(SearchAgents_no, 1);
		MatrixXd Xmax = ub.replicate(SearchAgents_no, 1);
		int elite_idx = dis(gen);
		MatrixXd Elite = Prey_old.row(elite_idx).replicate(SearchAgents_no, 1);

		double CF = pow(1.0 - static_cast<double>(Iter) / Max_iter, 2.0 * Iter / Max_iter);
		MatrixXd RL = 0.05 * mompa_levy(SearchAgents_no, dim, 1.5);
		MatrixXd RB = MatrixXd::NullaryExpr(SearchAgents_no, dim,
			[&]() { return normal_distribution<double>(0, 1)(gen); });
		MatrixXd stepsize = MatrixXd::Zero(SearchAgents_no, dim);

		for (int i = 0; i < Prey.rows(); i++) {      
			for (int j = 0; j < Prey.cols(); j++) {
				double R = int_dis(gen);
				if (Iter < Max_iter / 3) {
					stepsize(i, j) = RB(i, j) * (Elite(i, j) - RB(i, j) * Prey(i, j));
					Prey(i, j) = Prey(i, j) + P * R * stepsize(i, j);
				}
				else if (Iter > Max_iter/3 && Iter < 2 * Max_iter / 3) {
					if (i > Prey.rows() / 2) {
						stepsize(i, j) = RB(i, j) * (RB(i, j) * Elite(i, j) - Prey(i, j));
						Prey(i, j) = Elite(i, j) + P * CF * stepsize(i, j);
					}
					else {
						stepsize(i, j) = RL(i, j) * (Elite(i, j) - RL(i, j) * Prey(i, j));
						Prey(i, j) = Prey(i, j) + P * R * stepsize(i, j);
					}
				}
				else {
					stepsize(i, j) = RL(i, j) * (RL(i, j) * Elite(i, j) - Prey(i, j));
					Prey(i, j) = Elite(i, j) + P * CF * stepsize(i, j);
				}
			}
		}

		for (int i = 0; i < Prey.rows(); i++) {
			for (int j = 0; j < Prey.cols(); j++) {
				if (Prey(i, j) > ub(j)) {
					Prey(i, j) = ub(j);  // 超过上界，设为上界值
				}
				else if (Prey(i, j) < lb(j)) {
					Prey(i, j) = lb(j);  // 低于下界，设为下界值
				}
			}
		}

		MatrixXd Prey_evo = Prey;  // 保存原始种群
		if (dis(gen) < FADs) {
			// FADs效应
			MatrixXd U = MatrixXd::NullaryExpr(SearchAgents_no, dim,
				[&]() { return dis(gen) < FADs ? 1.0 : 0.0; });
			// 生成随机矩阵
			MatrixXd rand_matrix = MatrixXd::NullaryExpr(SearchAgents_no, dim,
				[&]() { return dis(gen); });
			// 计算范围
			MatrixXd range = Xmax - Xmin;
			// 计算随机跳跃位置
			MatrixXd random_jump = Xmin + rand_matrix.cwiseProduct(range);
			Prey = Prey + CF * random_jump.cwiseProduct(U);
		}
		else {
			// 随机游走
			double r = dis(gen);
			//todo
			VectorXi rand_indices1 = VectorXi::NullaryExpr(SearchAgents_no,
				[&]() { return std::rand() % SearchAgents_no; });
			VectorXi rand_indices2 = VectorXi::NullaryExpr(SearchAgents_no,
				[&]() { return std::rand() % SearchAgents_no; });

			MatrixXd stepsize = (FADs * (1 - r) + r) *
				(Prey(rand_indices1, all) - Prey(rand_indices2, all));

			Prey = Prey + stepsize;
		}

		for (int i = 0; i < Prey.rows(); i++) {
			for (int j = 0; j < Prey.cols(); j++) {
				if (Prey(i, j) > ub(j)) {
					Prey(i, j) = ub(j);  // 超过上界，设为上界值
				}
				else if (Prey(i, j) < lb(j)) {
					Prey(i, j) = lb(j);  // 低于下界，设为下界值
				}
			}
		}

		MatrixXd Prey_fads = Prey;

		// 高斯搜索
		MatrixXd Prey_gau = mompa_Gaussian_search(Prey, SearchAgents_no, dim,
			ub, lb);

		// 组合种群
		MatrixXd two_Prey(Prey_old.rows() + Prey_evo.rows() + Prey_fads.rows() + Prey_gau.rows(), dim);
		two_Prey << Prey_old, Prey_evo, Prey_fads, Prey_gau;
		// 去除重复行
		MatrixXd two_loc = unique_rows(two_Prey);
		// 评估合并种群
		auto [subfit, _, P_1] = mompa_getMOFcn(FUN, two_loc, numObj); 

		// 更新理想点 
		MatrixXd combined_fit(Zmin.rows() + subfit.rows(), numObj);
		combined_fit << Zmin, subfit;
		Zmin = combined_fit.colwise().minCoeff();

		// 环境选择
		Prey = EnvironmentalSelection(FUN, two_loc, SearchAgents_no, numObj, Z, Zmin);

		// 评估最终种群
		MatrixXd fit;
		std::tie(fit, std::ignore, std::ignore) = mompa_getMOFcn(FUN, Prey, numObj);

		// 更新历史种群
		Prey_old = Prey;

		// 计算IGD指标
		sub_IGD[Iter] = mompa_IGD(fit, P_1);

		Iter++;
		if (Iter % 100 == 0) {
			cout << "Iteration " << Iter << "/" << Max_iter << endl;
		}
 
	}
	return make_tuple(fit, sub_IGD, P_1);
} 

 
 
int main() {
	//int n = 9;
	//std::vector<Surface_mesh> parts(n);

	//for (int i = 0; i < n; i++) {
	//	string filedir=""
	//	string filename=""
	//	parts[i]=read_off_file()
	//}

	int runs = 1; // MATLAB 循环次数
	vector<double> a_mompa(runs, 0.0); 

	for (int i = 0; i < runs; ++i) {
		// ZDT3 参数
		//int numObj = 2;
		//int dim = 30;

		int numObj = 3;
		int dim = 12;
		int Max_iter = 1000;
		int SearchAgents_no = 100;

		RowVectorXd lb = RowVectorXd::Zero(dim);
		RowVectorXd ub = RowVectorXd::Ones(dim);
		// MATLAB: final_lb = zeros(1,12);
		//RowVectorXd lb = RowVectorXd::Zero(12);

		// MATLAB: final_ub = 2 : 2 : 2*12;
		//RowVectorXd ub = RowVectorXd::LinSpaced(12, 2, 24);

		//string F = "ZDT3";
		string F = "DTLZ4";

		auto start = std::chrono::high_resolution_clock::now();

		auto [fit, IGD, P] = mompa(
			F,
			SearchAgents_no,
			dim,
			lb,
			ub,
			numObj,
			Max_iter
		);

		// 结束计时
		auto end = std::chrono::high_resolution_clock::now();

		// 计算持续时间
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

		//std::cout << "ZDT3 Run " << i + 1 << " finished. IGD: " << IGD.back() << std::endl;
		if (!IGD.empty()) {
			a_mompa[i] = IGD.back();
		}
		
		std::cout << "执行时间: " << duration.count() / 1000.0 << " s" << std::endl;
	}

	// 输出平均值和标准差
	double mean_val = accumulate(a_mompa.begin(), a_mompa.end(), 0.0) / a_mompa.size();

	double sq_sum = 0.0;
	for (auto v : a_mompa) sq_sum += (v - mean_val) * (v - mean_val);
	double std_val = sqrt(sq_sum / a_mompa.size());

	//cout << "IGD mean: " << mean_val << ", std: " << std_val << endl;
	cout << "IGD mean: " << mean_val << endl;

	//// 输出参考 Pareto 前沿 P
	//cout << "Pareto front (P_1) sample points:" << endl;
	//cout << P.topRows(5) << endl; // 打印前5行作为示例

	// 可视化部分：如果想画3D散点图，可以用 matplotlib-cpp 或者输出 CSV 再 Python 绘图

	return 0;
}