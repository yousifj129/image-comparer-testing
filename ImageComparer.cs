using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Factorization;
using System.Drawing.Imaging;
namespace testing_stuff_console_place
{
    public class Histogram
    {
        public int[] Data { get; set; }
        public int Size { get; set; }
        public int MaxValue { get; set; }

        public Histogram(int width, int height)
        {
            Data = new int[width * height];
            Size = width * height;
            MaxValue = int.MinValue;
        }

        public void Add(int value)
        {
            Data[value]++;
            if (Data[value] > MaxValue)
            {
                MaxValue = Data[value];
            }
        }
    }
    public class ImageComparer
    {
        public static double CompareSSIM(Bitmap image1, Bitmap image2)
        {
            int width = Math.Min(image1.Width, image2.Width);
            int height = Math.Min(image1.Height, image2.Height);

            const double K1 = 0.01;
            const double K2 = 0.03;
            const double L = 255;

            double c1 = (K1 * L) * (K1 * L);
            double c2 = (K2 * L) * (K2 * L);

            double sumSSIM = 0.0;
            int numBlocks = 0;

            for (int y = 0; y < height; y += 8)
            {
                for (int x = 0; x < width; x += 8)
                {
                    double luminanceSum = 0.0;
                    double contrastSum = 0.0;
                    double structureSum = 0.0;

                    int blockWidth = Math.Min(8, width - x);
                    int blockHeight = Math.Min(8, height - y);

                    for (int offsetY = 0; offsetY < blockHeight; offsetY++)
                    {
                        for (int offsetX = 0; offsetX < blockWidth; offsetX++)
                        {
                            Color pixel1 = image1.GetPixel(x + offsetX, y + offsetY);
                            Color pixel2 = image2.GetPixel(x + offsetX, y + offsetY);

                            double luminance1 = (0.299 * pixel1.R) + (0.587 * pixel1.G) + (0.114 * pixel1.B);
                            double luminance2 = (0.299 * pixel2.R) + (0.587 * pixel2.G) + (0.114 * pixel2.B);

                            luminanceSum += luminance1 * luminance2;
                            contrastSum += (luminance1 - luminance2) * (luminance1 - luminance2);
                            structureSum += (pixel1.R - pixel2.R) * (pixel1.R - pixel2.R)
                                            + (pixel1.G - pixel2.G) * (pixel1.G - pixel2.G)
                                            + (pixel1.B - pixel2.B) * (pixel1.B - pixel2.B);
                        }
                    }

                    double luminanceAvg = luminanceSum / (blockWidth * blockHeight);
                    double contrastAvg = contrastSum / (blockWidth * blockHeight);
                    double structureAvg = structureSum / (blockWidth * blockHeight);

                    double ssim = ((2 * luminanceAvg + c1) * (2 * structureAvg + c2)) / ((luminanceAvg * luminanceAvg + contrastAvg + c1) * (structureAvg * structureAvg + c2));

                    sumSSIM += ssim;
                    numBlocks++;
                }
            }

            double avgSSIM = sumSSIM / numBlocks;

            return avgSSIM;
        }

        public static double CompareNCC(Bitmap image1, Bitmap image2)
        {
            if (image1.Width != image2.Width || image1.Height != image2.Height)
            {
                // Resize the images to have the same dimensions
                int maxWidth = Math.Min(image1.Width, image2.Width);
                int maxHeight = Math.Min(image1.Height, image2.Height);
                image1 = ResizeImage(image1, maxWidth, maxHeight);
                image2 = ResizeImage(image2, maxWidth, maxHeight);
            }
            // Convert the images to grayscale
            Bitmap grayImage1 = ConvertToGrayscale(image1);
            Bitmap grayImage2 = ConvertToGrayscale(image2);

            // Calculate the mean intensities
            double mean1 = CalculateMean(grayImage1);
            double mean2 = CalculateMean(grayImage2);

            // Calculate the standard deviations
            double stdDev1 = CalculateStandardDeviation(grayImage1, mean1);
            double stdDev2 = CalculateStandardDeviation(grayImage2, mean2);

            double sum = 0.0;

            // Calculate the NCC value
            for (int y = 0; y < grayImage1.Height; y++)
            {
                for (int x = 0; x < grayImage1.Width; x++)
                {
                    int pixel1 = grayImage1.GetPixel(x, y).R;
                    int pixel2 = grayImage2.GetPixel(x, y).R;

                    sum += (pixel1 - mean1) * (pixel2 - mean2);
                }
            }

            double ncc = sum / (grayImage1.Width * grayImage1.Height * stdDev1 * stdDev2);

            return ncc;
        }

        

        #region leven
        public static double CompareLeven(Bitmap image1, Bitmap image2)
        {
            int width = Math.Min(image1.Width, image2.Width);
            int height = Math.Min(image1.Height, image2.Height);

            Bitmap resizedImage1 = ResizeImage(image1, width, height);
            Bitmap resizedImage2 = ResizeImage(image2, width, height);

            string asciiImage1 = ConvertImageToASCII(resizedImage1);
            string asciiImage2 = ConvertImageToASCII(resizedImage2);

            int maxDistance = Math.Max(asciiImage1.Length, asciiImage2.Length);
            int similarityScore = CalculateLevenshteinDistance(asciiImage1, asciiImage2);

            double similarityRatio = 1.0 - (double)similarityScore / maxDistance;

            return similarityRatio;
        }
        private static int CalculateLevenshteinDistance(string str1, string str2)
        {
            int[] distances = new int[str2.Length + 1];

            for (int j = 0; j <= str2.Length; j++)
            {
                distances[j] = j;
            }

            for (int i = 1; i <= str1.Length; i++)
            {
                int previous = distances[0];
                distances[0] = i;

                for (int j = 1; j <= str2.Length; j++)
                {
                    int cost = (str1[i - 1] != str2[j - 1]) ? 1 : 0;
                    int insertDelete = Math.Min(distances[j] + 1, distances[j - 1] + 1);
                    int substitute = distances[j - 1] + cost;

                    int current = Math.Min(insertDelete, substitute);
                    distances[j - 1] = previous;
                    previous = current;
                }

                distances[str2.Length] = previous;
            }

            return distances[str2.Length];
        }
        #endregion


        #region histogram
        public static double CompareHistogram(Bitmap image1, Bitmap image2)
        {
            if (image1.Width != image2.Width || image1.Height != image2.Height)
            {
                // Resize the images to have the same dimensions
                int maxWidth = Math.Min(image1.Width, image2.Width);
                int maxHeight = Math.Min(image1.Height, image2.Height);
                image1 = ResizeImage(image1, maxWidth, maxHeight);
                image2 = ResizeImage(image2, maxWidth, maxHeight);
            }
            // Convert the images to grayscale
            Bitmap grayImage1 = ConvertToGrayscale(image1);
            Bitmap grayImage2 = ConvertToGrayscale(image2);

            // Get the histograms
            Histogram histogram1 = GetHistogram(grayImage1);
            Histogram histogram2 = GetHistogram(grayImage2);

            // Calculate the difference between the histograms
            float difference = CalculateHistogramDifference(histogram1, histogram2);

            difference /= (float)Math.Max(histogram1.MaxValue, histogram2.MaxValue);

            return difference;
        }
        #endregion

        #region helper functions
        static Histogram GetHistogram(Bitmap image)
        {
            // Get the histogram of the image
            Histogram histogram = new Histogram(image.Width, image.Height);

            // Iterate over each pixel in the image
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    // Get the pixel value
                    Color pixel = image.GetPixel(x, y);

                    // Add the pixel value to the histogram
                    histogram.Add(((int)(pixel.GetBrightness()*255f)));
                }
            }

            return histogram;
        }

        static float CalculateHistogramDifference(Histogram histogram1, Histogram histogram2)
        {
            // Calculate the difference between the histograms
            int difference = 0;
            for (int i = 0; i < histogram1.Size; i++)
            {
                int count1 = histogram1.Data[i];
                int count2 = histogram2.Data[i];
                difference += Math.Abs(count1 - count2);
            }

            return difference;
        }
        private static Bitmap ConvertToGrayscale(Bitmap image)
        {
            Bitmap grayImage = new Bitmap(image.Width, image.Height);

            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    Color pixel = image.GetPixel(x, y);
                    int grayValue = (int)(0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B);
                    grayImage.SetPixel(x, y, Color.FromArgb(grayValue, grayValue, grayValue));
                }
            }

            return grayImage;
        }

        private static double CalculateMean(Bitmap image)
        {
            double sum = 0.0;

            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    int pixel = image.GetPixel(x, y).R;
                    sum += pixel;
                }
            }

            double mean = sum / (image.Width * image.Height);

            return mean;
        }

        private static double CalculateStandardDeviation(Bitmap image, double mean)
        {
            double sum = 0.0;

            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    int pixel = image.GetPixel(x, y).R;
                    sum += (pixel - mean) * (pixel - mean);
                }
            }

            double variance = sum / (image.Width * image.Height);
            double stdDev = Math.Sqrt(variance);

            return stdDev;
        }
        public static Bitmap ResizeImage(Bitmap image, int width, int height)
        {
            Bitmap resizedImage = new Bitmap(width, height);
            using (Graphics graphics = Graphics.FromImage(resizedImage))
            {
                graphics.DrawImage(image, 0, 0, width, height);
            }
            return resizedImage;
        }
        public static string ConvertImageToASCII(Bitmap image)
        {
            StringBuilder sb = new StringBuilder();

            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    Color pixelColor = image.GetPixel(x, y);
                    byte brightness = (byte)((pixelColor.R + pixelColor.G + pixelColor.B) / 3);
                    char character = GetASCIICharacter(brightness);
                    sb.Append(character);
                }
                sb.Append(Environment.NewLine);
            }

            return sb.ToString();
        }

        public static char GetASCIICharacter(byte brightness)
        {
            // Map the brightness value to an ASCII character based on some predefined threshold values
            // You can customize this mapping according to your requirements
            if (brightness >= 230)
            {
                return ' ';
            }
            else if (brightness >= 200)
            {
                return '.';
            }
            else if (brightness >= 180)
            {
                return '-';
            }
            else if (brightness >= 150)
            {
                return '=';
            }
            else if (brightness >= 120)
            {
                return '+';
            }
            else if (brightness >= 80)
            {
                return '*';
            }
            else if (brightness >= 50)
            {
                return '#';
            }
            else
            {
                return '@';
            }
        }
        #endregion
    }

    public class EigenvalueDecomposition
    {
        private Matrix<double> matrix;
        private MathNet.Numerics.LinearAlgebra.Vector<double> eigenvalues;
        private Matrix<double> eigenvectors;

        public EigenvalueDecomposition(double[,] matrix)
        {
            this.matrix = DenseMatrix.OfArray(matrix);
            CalculateEigenvalueDecomposition();
        }

        public double[] Eigenvalues
        {
            get { return eigenvalues.ToArray(); }
        }

        public double[,] Eigenvectors
        {
            get { return eigenvectors.ToArray(); }
        }

        public double[,] GetV()
        {
            return eigenvectors.ToArray();
        }

        private void CalculateEigenvalueDecomposition()
        {
            Evd<double> eigDecomposition = matrix.Evd(Symmetricity.Symmetric);
            eigenvalues = eigDecomposition.EigenValues.Real();
            eigenvectors = eigDecomposition.EigenVectors;
        }
    }
    public class Eigenfaces
    {
        private List<double[]> trainingData; // List of training face images
        private double[,] eigenVectors; // Eigenvectors of the covariance matrix
        private double[] averageFace; // Average face vector
        private int numEigenfaces; // Number of eigenfaces to keep

        public Eigenfaces(int numEigenfaces)
        {
            this.numEigenfaces = numEigenfaces;
        }

        public void Train(List<Bitmap> trainingImages)
        {
            int numTrainingImages = trainingImages.Count;
            int minImageWidth = int.MaxValue;
            int minImageHeight = int.MaxValue;

            // Find the minimum width and height among the training images
            foreach (Bitmap image in trainingImages)
            {
                if (image.Width < minImageWidth)
                    minImageWidth = image.Width;

                if (image.Height < minImageHeight)
                    minImageHeight = image.Height;
            }

            // Convert training images to grayscale and flatten them into vectors
            trainingData = new List<double[]>();
            foreach (Bitmap image in trainingImages)
            {
                Bitmap resizedImage = ImageComparer.ResizeImage(image, minImageWidth, minImageHeight); // Resize image to the minimum dimensions
                Bitmap grayImage = ConvertToGrayscale(resizedImage);
                double[] faceVector = FlattenImage(grayImage, minImageWidth, minImageHeight);
                trainingData.Add(faceVector);
            }

            // Calculate the average face vector
            averageFace = CalculateAverageFace(trainingData);

            // Subtract the average face from each training face vector
            List<double[]> normalizedData = NormalizeTrainingData(trainingData, averageFace);

            // Construct the covariance matrix
            double[,] covarianceMatrix = CalculateCovarianceMatrix(normalizedData);

            // Perform eigenvalue decomposition on the covariance matrix
            EigenvalueDecomposition eigDecomposition = new EigenvalueDecomposition(covarianceMatrix);
            eigenVectors = eigDecomposition.GetV();

            // Keep only the top eigenfaces
            eigenVectors = SelectTopEigenfaces(eigenVectors, numEigenfaces);
        }

        public string Recognize(Bitmap inputImage)
        {
            // Convert input image to grayscale and flatten it into a vector
            Bitmap grayInputImage = ConvertToGrayscale(inputImage);
            double[] inputFace = FlattenImage(grayInputImage, inputImage.Width, inputImage.Height);

            // Subtract the average face from the input face vector
            double[] normalizedInputFace = SubtractAverageFace(inputFace, averageFace);

            // Project the normalized input face onto the eigenfaces
            double[] inputWeights = ProjectOntoEigenfaces(normalizedInputFace, eigenVectors);

            // Compare input weights with training weights to find the closest match
            double minDistance = double.MaxValue;
            string bestMatch = "";

            foreach (double[] trainingFace in trainingData)
            {
                double[] trainingWeights = ProjectOntoEigenfaces(trainingFace, eigenVectors);
                double distance = CalculateEuclideanDistance(inputWeights, trainingWeights);

                if (distance < minDistance)
                {
                    minDistance = distance;
                    bestMatch = "Match found!"; // Replace this with the label/identifier of the training image
                }
            }

            return bestMatch;
        }

        private Bitmap ConvertToGrayscale(Bitmap image)
        {
            Bitmap grayImage = new Bitmap(image.Width, image.Height);

            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    Color pixel = image.GetPixel(x, y);
                    int grayValue = (int)(0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B);
                    grayImage.SetPixel(x, y, Color.FromArgb(grayValue, grayValue, grayValue));
                }
            }

            return grayImage;
        }

        private double[] FlattenImage(Bitmap image, int width, int height)
        {
            double[] vector = new double[width * height];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Color pixel = image.GetPixel(x, y);
                    int grayValue = pixel.R;
                    vector[y * width + x] = grayValue;
                }
            }

            return vector;
        }

        private double[] CalculateAverageFace(List<double[]> trainingData)
        {
            int numImages = trainingData.Count;
            int vectorLength = trainingData[0].Length;
            double[] averageVector = new double[vectorLength];

            for (int i = 0; i < numImages; i++)
            {
                for (int j = 0; j < vectorLength; j++)
                {
                    averageVector[j] += trainingData[i][j];
                }
            }

            for (int j = 0; j < vectorLength; j++)
            {
                averageVector[j] /= numImages;
            }

            return averageVector;
        }

        private List<double[]> NormalizeTrainingData(List<double[]> trainingData, double[] averageFace)
        {
            int numImages = trainingData.Count;
            int vectorLength = trainingData[0].Length;

            List<double[]> normalizedData = new List<double[]>();

            for (int i = 0; i < numImages; i++)
            {
                double[] normalizedVector = SubtractAverageFace(trainingData[i], averageFace);
                normalizedData.Add(normalizedVector);
            }

            return normalizedData;
        }

        private double[,] CalculateCovarianceMatrix(List<double[]> data)
        {
            int numImages = data.Count;
            long vectorLength = data[0].LongLength;

            double[,] covarianceMatrix = new double[vectorLength, vectorLength];

            for (int i = 0; i < numImages; i++)
            {
                double[] vector = data[i];

                for (int j = 0; j < vectorLength; j++)
                {
                    for (int k = 0; k < vectorLength; k++)
                    {
                        covarianceMatrix[j, k] += (vector[j] * vector[k]);
                    }
                }
            }

            for (int j = 0; j < vectorLength; j++)
            {
                for (int k = 0; k < vectorLength; k++)
                {
                    covarianceMatrix[j, k] /= numImages;
                }
            }

            return covarianceMatrix;
        }

        private double[,] SelectTopEigenfaces(double[,] eigenVectors, int numEigenfaces)
        {
            int vectorLength = eigenVectors.GetLength(0);
            double[,] topEigenVectors = new double[numEigenfaces, vectorLength];

            for (int i = 0; i < numEigenfaces; i++)
            {
                for (int j = 0; j < vectorLength; j++)
                {
                    topEigenVectors[i, j] = eigenVectors[i, j];
                }
            }

            return topEigenVectors;
        }

        private double[] SubtractAverageFace(double[] inputFace, double[] averageFace)
        {
            int vectorLength = inputFace.Length;
            double[] result = new double[vectorLength];

            for (int i = 0; i < vectorLength; i++)
            {
                result[i] = inputFace[i] - averageFace[i];
            }

            return result;
        }

        private double[] ProjectOntoEigenfaces(double[] inputFace, double[,] eigenVectors)
        {
            int numEigenfaces = eigenVectors.GetLength(0);
            int vectorLength = inputFace.Length;
            double[] weights = new double[numEigenfaces];

            for (int i = 0; i < numEigenfaces; i++)
            {
                for (int j = 0; j < vectorLength; j++)
                {
                    weights[i] += eigenVectors[i, j] * inputFace[j];
                }
            }

            return weights;
        }

        private double CalculateEuclideanDistance(double[] vector1, double[] vector2)
        {
            int vectorLength = vector1.Length;
            double distance = 0.0;

            for (int i = 0; i < vectorLength; i++)
            {
                distance += Math.Pow(vector1[i] - vector2[i], 2);
            }

            return Math.Sqrt(distance);
        }
    }
}
