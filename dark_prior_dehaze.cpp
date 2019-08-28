#include<opencv2/opencv.hpp>
#include<iostream>
#include<iomanip>
#include<string>
#include<vector>
#include<ctime>
#include<algorithm>
using namespace std;
using namespace cv;
string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}
bool isU(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  return true; break;
	case CV_8S:  return false; break;
	case CV_16U: return true; break;
	case CV_16S: return false; break;
	case CV_32S: return false; break;
	case CV_32F: return false;  break;
	case CV_64F: return false; break;
	default:     return false; break;
	}
}
bool isF(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  return false; break;
	case CV_8S:  return false; break;
	case CV_16U: return false; break;
	case CV_16S: return false; break;
	case CV_32S: return false; break;
	case CV_32F: return true;  break;
	case CV_64F: return true; break;
	default:     return false; break;
	}
}
Mat U2F(Mat src)
{
	// convert unsigned char (0-255) image to float (0-1) image
	Mat dst;
	src.convertTo(dst, CV_32F, 1 / 255.0);
	return dst;
}
Mat F2U(Mat src)
{
	// convert float (0-1) image to unsigned char (0-255) image
	Mat dst = src.clone();
	src.convertTo(dst, CV_8U, 255);
	return dst;
}
void print_size(Mat src)
{
	// print height width channels
	int size_h = src.size[0];
	int size_w = src.size[1];
	int size_z = src.channels();
	cout << type2str(src.type()) << endl;
	cout << "h: " << size_h << " w: " << size_w << " z: " << size_z << endl;
	return;
}
double min(Mat Y)
{
	// return min element in an image
	double min, max;
	minMaxIdx(Y, &min, &max);
	return min;
}
double max(Mat Y)
{
	// return max element in an image
	double min, max;
	minMaxIdx(Y, &min, &max);
	return max;
}
void arg_min(Mat Y, int idx[2])
{
	minMaxIdx(Y, 0, 0, idx, 0);
	return;
}
void arg_max(Mat Y, int idx[2])
{
	minMaxIdx(Y, 0, 0, 0, idx);
	return;
}
Mat reduce_min(Mat in)
{
	Mat src = in.clone();
	
	// reduce min of an image over the third dimension
	int size_h = src.size[0];
	int size_w = src.size[1];
	int size_z = src.channels();

	vector<Mat> channels(size_z);
	split(src, channels);
	Mat minmat;
	if (isF(src.type()))
	{
		minmat = Mat(size_h, size_w, CV_32F, Scalar(FLT_MAX));
	}
	else if (isU(src.type()))
	{
		minmat = Mat(size_h, size_w, CV_8U, Scalar(255));
	}
	
	for (int z = 0; z < size_z; ++z)
	{
		min(channels[z], minmat, minmat);
	}

	return minmat;
}
Mat reduce_max(Mat in)
{
	Mat src = in.clone();
	
	// reduce max of an image over the third dimension
	int size_h = src.size[0];
	int size_w = src.size[1];
	int size_z = src.channels();

	vector<Mat> channels(size_z);
	split(src, channels);
	Mat maxmat;
	if (isF(src.type()))
	{
		maxmat = Mat(size_h, size_w, CV_32F, Scalar(FLT_MIN));
	}
	else if (isU(src.type()))
	{
		maxmat = Mat(size_h, size_w, CV_8U, Scalar(0));
	}
	
	for (int z = 0; z < size_z; ++z)
	{
		max(channels[z], maxmat, maxmat);
	}

	return maxmat;
}
Mat reduce_mean(Mat in)
{
	Mat src = in.clone();
	if (!isF(src.type()))
	{
		src = U2F(src);
	}
	// reduce mean of an image over the third dimension
	int size_h = src.size[0];
	int size_w = src.size[1];
	int size_z = src.channels();

	vector<Mat> channels(size_z);

	split(src, channels);
	Mat summat(size_h, size_w, CV_32F, Scalar(0));
	for (int z = 0; z < size_z; ++z)
	{
		summat = summat + channels[z];
	}
	return summat/size_z;
}
Mat cumsum(Mat src)
{
	Mat dst = src.clone();
	if (!isF(dst.type()))
	{
		dst = U2F(dst);
	}
	// cumsum of Mat array
	int size_h = dst.size[0];
	int size_w = dst.size[1];
	
	for (int i = 1; i < size_h; ++i)
	{
		dst.at<float>(i) += dst.at<float>(i - 1);
	}
	return dst;
}
template<typename T>
vector<float> cumsum_vec(vector<T> src)
{
	vector<float> dst(src.size(), 0);
	for (int i = 0; i < src.size(); ++i)
	{
		dst[i] = (float)src[i];
	}
	for (int i = 1; i < src.size(); ++i)
	{
		dst[i] += dst[i - 1];
	}
	return dst;
}
Mat myCalcHist(Mat V1, int bins = 100)
{
	Mat ht;
	int channels[] = { 0 };
	int histSize[] = { bins };
	float granges[] = { 0, 255 };
	const float* ranges[] = { granges };
	calcHist(&V1, 1, channels, Mat(), ht, 1, histSize, ranges, true, false);
	return ht;

}
bool all(Mat src)
{
	int size_h = src.size[0];
	int size_w = src.size[1];
	int size_z = src.channels();
	bool result = true;
	for (int i = 0; i < size_h; i++)
	{
		for (int j = 0; j < size_w; j++)
		{
			for (int z = 0; z < size_z; z++)
			{
				if (src.at<uchar>(i,j,z) == 0)
					result = false;
			}
		}
	}
	return result;
}
bool all(vector<bool> mask)
{
	bool result = true;
	for (int i = 0; i < mask.size(); i++)
	{
		if (!mask[i])
			result = false;
	}
	return result;
}
template<typename T>
vector<T> abs(vector<T> src)
{
	vector<T> dst(src.begin(), src.end());
	for (int i = 0; i < src.size(); i++)
	{
		dst[i] = abs(src[i]);
	}
	return dst;
}
Mat masked_array(Mat src, Mat mask, int padd=255)
{
	int size_h = src.size[0];
	int size_w = src.size[1];
	int size_z = src.channels();
	Mat result = src.clone();
	for (int i = 0; i < size_h; i++)
	{
		for (int j = 0; j < size_w; j++)
		{
			for (int z = 0; z < size_z; z++)
			{
				if (!(mask.at<uchar>(i, j, z) == 0))
				{
					result.at<int>(i, j, z) = padd;
				}
					
				else
				{
					result.at<int>(i, j, z) = src.at<int>(i, j, z);
				}
					
			}
		}
	}
	return result;
}
template<typename T>
vector<T> masked_array(vector<T> src, vector<bool> mask, T padd)
{
	vector<T> result(src.size(), padd);
	for (int i = 0; i < src.size(); i++)
	{
		if (!(mask[i] == false))
		{
			result[i] = padd;
		}

		else
		{
			result[i] = src[i];
		}
	}
	return result;
}
template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(
	const std::vector<T>& vec,
	Compare& compare)
{
	std::vector<std::size_t> p(vec.size());
	std::iota(p.begin(), p.end(), 0);
	std::sort(p.begin(), p.end(),
		[&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
	return p;
}
template <typename T>
std::vector<T> apply_permutation(
	const std::vector<T>& vec,
	const std::vector<std::size_t>& p)
{
	std::vector<T> sorted_vec(vec.size());
	std::transform(p.begin(), p.end(), sorted_vec.begin(),
		[&](std::size_t i){ return vec[i]; });
	return sorted_vec;
}

void unique(vector<int> src, vector<int>& uni, vector<int>& idx, vector<int>& count)
{
	idx = vector<int>(src.size(), 0);
	int hash[256] = { 0 };
	for (int i = 0; i < src.size(); i++)
	{
		hash[src[i]]++;
	}
	int k = 0;
	int new_idx[256];
	for (int i = 0; i < 256; i++)
	{
		if (hash[i] != 0)
		{
			uni.push_back(i);
			count.push_back(hash[i]);
			new_idx[i] = k;
			k++;
		}
	}
	for (int i = 0; i < src.size(); i++)
	{
		idx[i] = new_idx[src[i]];
	}
	
	return;
}
template<typename T>
void print(std::vector <T> const &a) {
	std::cout << "The vector elements are : ";

	for (int i = 0; i < a.size(); i++)
		std::cout << a.at(i) << ' ';
	std::cout << "\n" << endl;
}
template<typename T>
std::vector<T> slice_vec(std::vector<T> const &v, int m, int n)
{
	auto first = v.cbegin() + m;
	auto last = v.cbegin() + n + 1;

	std::vector<T> vec(first, last);
	return vec;
}


Mat zmMinFilterGray(Mat src, int r = 7)
{
	Mat dst;
	erode(src, dst, getStructuringElement(MORPH_RECT, Size(2 * r + 1, 2 * r + 1)));

	return dst;
}
Mat guidedfilter(Mat I, Mat p, int r, float eps)
{
	Mat m_I, m_p, m_Ip, m_II, m_a, m_b;
	boxFilter(I, m_I, -1, { r, r });
	boxFilter(p, m_p, -1, { r, r });
	boxFilter(I.mul(p), m_Ip, -1, { r, r });
	Mat cov_Ip = m_Ip - m_I.mul(m_p);
	boxFilter(I.mul(I), m_II, -1, { r, r });
	Mat var_I = m_II - m_I.mul(m_I);
	Mat a = cov_Ip / (var_I + eps);
	Mat b = m_p - a.mul(m_I);
	boxFilter(a, m_a, -1, { r, r });
	boxFilter(b, m_b, -1, { r, r });
	return m_a.mul(I) + m_b;
}
int getV1(Mat m, int r, float eps, float w, float maxV1, Mat &V1, float& A)
{
	V1 = reduce_min(m);
	V1 = guidedfilter(V1, zmMinFilterGray(V1, 7), r, eps);
	int bins = 100;
	Mat ht = myCalcHist(V1, bins);

	ht.convertTo(ht, CV_32F, V1.size[0] * V1.size[1]);
	Mat d = cumsum(ht);
	int lmax = bins - 1;
	for (; lmax > 0; lmax--)
	{
		if (d.at<float>(lmax) < 0.999)
			break;
	}
	Mat avg = reduce_mean(m);
	int size_h = m.size[0];
	int size_w = m.size[1];

	A = -1;                 //negative inf
	for (int i = 0; i < size_h; ++i)
	{
		for (int j = 0; j < size_w; ++j)
		{
			if (V1.at<float>(i, j) >= lmax / bins)
				if (avg.at<float>(i, j)>A)
					A = avg.at<float>(i, j);
		}
	}
	min(V1*w, maxV1, V1);
	return 0;
}
Mat deHaze(Mat m, int r = 81, float eps = 0.001, float w = 0.95, float maxV1 = 0.80)
{
	if (!isF(m.type()))
	{
		m = U2F(m);
	}
	int size_h = m.size[0];
	int size_w = m.size[1];
	int size_z = m.channels();
	Mat Y = m.clone();
	Mat V1;
	float A;
	getV1(m, r, eps, w, maxV1, V1, A);
	waitKey(0);
	vector<Mat> channels_Y(size_z);
	vector<Mat> channels_m(size_z);
	split(m, channels_m);
	for (int k = 0; k < size_z; ++k)
	{
		channels_Y[k] = (channels_m[k] - V1) / (1 - V1 / A);
	}
	merge(channels_Y, Y);
	max(Y, 0, Y); //clip 
	min(Y, 1, Y); //clip

	return Y;
}
void normalize(Mat &src)
{
	src.convertTo(src, CV_32F, 1.0/max(src));
	src.convertTo(src, CV_8U, 255);
}
template<typename T>
int find_nearest_above(vector<T> my_array, T target)
{
	vector<T> diff(my_array.size(), 0);
	for (int i = 0; i < my_array.size(); i++)
	{
		diff[i] = my_array[i] - target;
	}
	vector<bool> mask(my_array.size(), false);
	for (int i = 0; i < diff.size(); i++)
	{
		mask[i] = diff[i]<=-1;
	}
	if (all(mask))
	{
		vector<T> abs_diff = abs(diff);
		vector<T>::iterator iter = min_element(abs_diff.begin(), abs_diff.end());
		int c = iter - abs_diff.begin();
		return c;
	}
	
	vector<T> masked_diff = masked_array(diff, mask, 255);


	vector<T>::iterator iter = min_element(masked_diff.begin(), masked_diff.end());
	int c = iter - masked_diff.begin();
	return c;
}
Mat hist_match(Mat original, Mat specified)
{

	int ori_h = original.size[0];
	int ori_w = original.size[1];
	int sp_h = specified.size[0];
	int sp_w = specified.size[1];
	vector<int> ori_array(ori_h*ori_w, 0), sp_array(sp_h*sp_w,0);
	for (int i = 0; i < ori_h; i++)
	{
		for (int j = 0; j < ori_w; j++)
		{
			ori_array[i*ori_w + j] = original.at<uchar>(i, j);
		}
	}
	for (int i = 0; i < sp_h; i++)
	{
		for (int j = 0; j < sp_w; j++)
		{
			sp_array[i*sp_w + j] = specified.at<uchar>(i, j);
		}
	}
	vector<int> s_values, bin_idx, s_counts, t_values, t_counts;
	unique(ori_array, s_values, bin_idx, s_counts);
	unique(sp_array, t_values, vector<int>(), t_counts);
	vector<float> s_quantities = cumsum_vec(s_counts);
	vector<float> t_quantities = cumsum_vec(t_counts);
	for (int i = 0; i < s_quantities.size(); i++)
	{
		s_quantities[i] /= s_quantities[s_quantities.size()-1];
	}
	for (int i = 0; i < t_quantities.size(); i++)
	{
		t_quantities[i] /= t_quantities[t_quantities.size() - 1];
	}
	vector<int> sour(s_quantities.size(), 0), temp(t_quantities.size(), 0);
	for (int i = 0; i < s_quantities.size(); i++)
	{
		sour[i] = (int)(s_quantities[i] * 255);
	}
	for (int i = 0; i < t_quantities.size(); i++)
	{
		temp[i] = (int)(t_quantities[i] * 255);
	}
	vector<int> b;
	for (int i = 0; i < sour.size(); i++)
	{
		b.push_back(find_nearest_above(temp, sour[i]));
	}

	Mat dst = original.clone();
	int k = 0;
	for (int i = 0; i < ori_h; i++)
	{
		for (int j = 0; j < ori_w; j++)
		{
			dst.at<uchar>(i, j) = b[bin_idx[k]];
			k++;
		}
	}
	
	return dst;
}
Mat bright_hist_match(Mat src)
{
	if (!isU(src.type()))
	{
		src = F2U(src);
	}
	
	int size_h = src.size[0];
	int size_w = src.size[1];
	int size_z = src.channels();
	Mat dst;
	Mat equalized;
	
	vector<Mat> channels_src(size_z);
	vector<Mat> channels_tmp(size_z);
	vector<Mat> channels_dst(size_z);
	split(src, channels_src);

	
	for (int i = 0; i < size_z; i++)
	{
		equalizeHist(channels_src[i], channels_tmp[i]);
	}
	merge(channels_tmp, equalized);
	Mat bright_prior = reduce_max(equalized);

	for (int i = 0; i < size_z; i++)
	{
		channels_dst[i] = hist_match(channels_tmp[i], bright_prior);
	}
	merge(channels_dst, dst);
	print_size(dst);
	normalize(dst);

	return dst;
}
int main()
{
	clock_t start, end;
	Mat img = imread("100.png");
	start = clock();
	img = bright_hist_match(img);
	end = clock();
	double cost = (double)(end - start)/CLOCKS_PER_SEC;
	cout << "使用时间：" << cost << "s" << endl;
	cout << "分辨率：" << img.size[1] << "x" << img.size[0] << endl;
	cout << type2str(img.type()) << endl;
	//cout << img << endl;

	imshow("test",img);
	waitKey();
	imwrite("test.png", img);
	waitKey();

	return 0;
}
