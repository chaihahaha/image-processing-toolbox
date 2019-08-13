#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<iostream>
#include<iomanip>
#include<string>
#include<vector>
#include<ctime>
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
	Mat dst = src.clone()*255;
	src.convertTo(dst, CV_8U, 1);
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
Mat reduce_min(Mat src)
{
	// reduce min of an image over the third dimension
	int size_h = src.size[0];
	int size_w = src.size[1];
	int size_z = src.channels();

	vector<Mat> channels(size_z);
	
	split(src, channels);
	Mat minmat(size_h, size_w, CV_32F, Scalar(DBL_MAX));
	for (int z = 0; z < size_z; ++z)
	{
		min(channels[z], minmat, minmat);
	}
	return minmat;
}
Mat reduce_mean(Mat src)
{
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
	// cumsum of Mat array
	Mat dst = src.clone();
	int size_h = dst.size[0];
	int size_w = dst.size[1];
	for (int i = 1; i < size_h; ++i)
	{
		dst.at<float>(i) += dst.at<float>(i - 1);
	}
	return dst;
}
Mat zmMinFilterGray(Mat src, int r = 7)
{
	Mat dst;
	erode(src, dst, getStructuringElement(MORPH_RECT, Size(2 * r + 1, 2 * r + 1)));
	//imshow("erode", dst);
	waitKey();

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
	Mat ht;
	V1 = reduce_min(m);
	//imshow("dark", V1);
	waitKey();

	V1 = guidedfilter(V1, zmMinFilterGray(V1, 7), r, eps);
	//imshow("filter", V1);
	waitKey();

	int bins = 100;

	int channels[] = { 0};
	int histSize[] = { bins };
	float granges[] = { 0, 1 };
	const float* ranges[] = { granges };
	calcHist(&V1, 1, channels, Mat(), ht, 1, histSize, ranges, true, false);
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

	//cout << "lmax" << lmax << endl;
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
	m = U2F(m);
	int size_h = m.size[0];
	int size_w = m.size[1];
	int size_z = m.channels();
	Mat Y = m.clone();
	Mat V1;
	float A;
	getV1(m, r, eps, w, maxV1, V1, A);
	//imshow("V1", V1);
	waitKey(0);
	vector<Mat> channels_Y(size_z);
	vector<Mat> channels_m(size_z);
	//split(Y, channels_Y);
	split(m, channels_m);
	//cout << "A" << A << endl;
	for (int k = 0; k < size_z; ++k)
	{
		channels_Y[k] = (channels_m[k] - V1) / (1 - V1 / A);
	}
	merge(channels_Y, Y);
	max(Y, 0, Y); //clip 
	min(Y, 1, Y); //clip

	Y = 255 * Y;
	Y = F2U(Y);
	return Y;
}
int main()
{
	clock_t start, end;
	Mat img = imread("100.png");
	start = clock();
	img = deHaze(img);
	end = clock();
	double cost = (double)(end - start)/CLOCKS_PER_SEC;
	//cout.setf(ios::fixed);
	cout << "使用时间：" << cost << "s" << endl;
	cout << "分辨率：" << img.size[1] << "x" << img.size[0] << endl;
	imshow("fuck",img);
	waitKey(0);
}
