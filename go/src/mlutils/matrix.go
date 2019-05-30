package mlutils

import (
	"errors"
	"math"
)

func MatAddFloat32(mat1 [][]float32, mat2 [][]float32) ([][]float32, error) {
	if len(mat1) != len(mat2) && len(mat1[0]) != len(mat2[0]) {
		return nil, errors.New("invalid matrix dimensions for matrix addition")
	}

	result := make([][]float32, len(mat1))
	for i := 0; i < len(mat1); i++ {
		result[i] = make([]float32, len(mat1[0]))
		for j := 0; j < len(mat1[i]); j++ {
			result[i][j] = mat1[i][j] + mat2[i][j]
		}
	}
	return result, nil
}

func MatSubtractFloat32(mat1 [][]float32, mat2 [][]float32) [][]float32 {

	result := make([][]float32, len(mat1))
	for i := 0; i < len(mat1); i++ {
		result[i] = make([]float32, len(mat1[0]))
		for j := 0; j < len(mat1[i]); j++ {
			result[i][j] = mat1[i][j] - mat2[i][j]
		}
	}
	return result
}

func MatSumFloat32(mat [][]float32) float32 {
	sum := float32(0.0)
	for i := 0; i < len(mat); i++ {
		for j := 0; j < len(mat[i]); j++ {
			sum += mat[i][j]
		}
	}
	return sum
}

func MatMultFloat32(mat1 [][]float32, mat2 [][]float32) [][]float32 {

	result := make([][]float32, len(mat1))
	for i := 0; i < len(mat1); i++ {
		result[i] = make([]float32, len(mat2[0]))
		for j := 0; j < len(mat2[0]); j++ {
			for k := 0; k < len(mat2); k++ {
				result[i][j] += mat1[i][k] * mat2[k][j]
			}
		}
	}
	return result
}

func Multiply(vec1, vec2 []float32) []float32 {
	result := make([]float32, len(vec1))
	for i, _ := range vec1 {
		result[i] = vec1[i] * vec2[i]
	}
	return result
}

func ScalarMatMultFloat32(scalar float32, mat [][]float32) [][]float32 {
	result := make([][]float32, len(mat))

	for i := 0; i < len(mat); i++ {
		result[i] = make([]float32, len(mat[i]))
		for j := 0; j < len(mat[i]); j++ {
			result[i][j] = scalar * mat[i][j]
		}
	}
	return result
}

func ScalarMatDivFloat32(scalar float32, mat [][]float32) [][]float32 {
	result := make([][]float32, len(mat))

	for i := 0; i < len(mat); i++ {
		result[i] = make([]float32, len(mat[i]))
		for j := 0; j < len(mat[i]); j++ {
			result[i][j] = mat[i][j] / scalar
		}
	}
	return result
}

func MatPlusScalarFloat32(mat [][]float32, scalar float32) [][]float32 {
	result := make([][]float32, len(mat))

	for i := 0; i < len(mat); i++ {
		result[i] = make([]float32, len(mat[i]))
		for j := 0; j < len(mat[i]); j++ {
			result[i][j] = mat[i][j] + scalar
		}
	}
	return result
}

func MatMinusScalarFloat32(mat [][]float32, scalar float32) [][]float32 {
	result := make([][]float32, len(mat))

	for i := 0; i < len(mat); i++ {
		result[i] = make([]float32, len(mat[i]))
		for j := 0; j < len(mat[i]); j++ {
			result[i][j] = mat[i][j] - scalar
		}
	}
	return result
}

func Transpose(mat [][]float32) [][]float32 {
	result := make([][]float32, len(mat[0]))
	for i := 0; i < len(mat); i++ {
		for j := 0; j < len(mat[0]); j++ {
			result[j] = append(result[j], mat[i][j])
		}
	}
	return result
}

func MeanFloat32(x []float32) float32 {
	sum := SumFloat32(x)
	return sum / float32(len(x))
}

func StandardDevFloat32(x []float32) float32 {
	n := float32(len(x))
	xBar := MeanFloat32(x)
	numerator := make([]float32, len(x))

	for i, val := range x {
		numerator[i] = (val - xBar) * (val - xBar)
	}
	return float32(math.Sqrt(float64(SumFloat32(numerator) / n)))
}

func SumFloat32(x []float32) float32 {
	sum := float32(0)
	for _, val := range x {
		sum += val
	}
	return sum
}

func R2(x []float32, y []float32) float32 {
	n := float32(len(x))
	meanX := MeanFloat32(x)
	meanY := MeanFloat32(y)
	deltaX := StandardDevFloat32(x)
	deltaY := StandardDevFloat32(y)
	summation := make([]float32, len(x))
	for i, _ := range x {
		summation[i] = (x[i] - meanX) * (y[i] - meanY)
	}
	r := (float32(1) / n) * SumFloat32(summation) / (deltaX * deltaY)
	return r * r
}

func MatMult1D(matA, matB []float64, mat1Rows, mat2Cols int) []float32 {
	matC := make([]float64, mat1Rows*mat2Cols)
	for (i=0; i<mat1Rows; i++) {
		for (j=0; j<mat2Cols; j++) {
			for (k=0; k<mat2Cols; k++) {
				matC[i*n+j] += matA[i*n+k]*matB[k*n+j];
			}
		}
	}
	return result	
}