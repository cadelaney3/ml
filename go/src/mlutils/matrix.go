package mlutils

import (
	"log"
)

// OnesMatFloat32 creates a rows*cols matrix of all ones of type float32
func OnesMatFloat32(rows, cols int) [][]float32 {
	mat := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		mat[i] = make([]float32, cols)
		for j := 0; j < cols; j++ {
			mat[i][j] = float32(1)
		}
	}
	return mat
}

// OnesMat creates a rows*cols matrix of all ones of type float64
func OnesMat(rows, cols int) [][]float64 {
	mat := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		mat[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			mat[i][j] = float64(1)
		}
	}
	return mat
}

// ZerosMatFloat32 creates a rows*cols matrix of all zeros of type float64
func ZerosMatFloat32(rows, cols int) [][]float32 {
	mat := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		mat[i] = make([]float32, cols)
		for j := 0; j < cols; j++ {
			mat[i][j] = float32(0)
		}
	}
	return mat
}

// ZerosMat creates a rows*cols matrix of all zeros of type float64
func ZerosMat(rows, cols int) [][]float64 {
	mat := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		mat[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			mat[i][j] = float64(0)
		}
	}
	return mat
}

// MatAddFloat32 adds two matrices (2D slices) of type float32
func MatAddFloat32(mat1, mat2 [][]float32) [][]float32 {
	if len(mat1) != len(mat2) && len(mat1[0]) != len(mat2[0]) {
		log.Fatal("Invalid matrix dimensions")
	}

	result := make([][]float32, len(mat1))
	for i := 0; i < len(mat1); i++ {
		result[i] = make([]float32, len(mat1[0]))
		for j := 0; j < len(mat1[i]); j++ {
			result[i][j] = mat1[i][j] + mat2[i][j]
		}
	}
	return result
}

// MatAdd adds two matrices (2D slices) of type float64
func MatAdd(mat1, mat2 [][]float64) [][]float64 {
	if len(mat1) != len(mat2) && len(mat1[0]) != len(mat2[0]) {
		log.Fatal("Invalid matrix dimensions")
	}

	result := make([][]float64, len(mat1))
	for i := 0; i < len(mat1); i++ {
		result[i] = make([]float64, len(mat1[0]))
		for j := 0; j < len(mat1[i]); j++ {
			result[i][j] = mat1[i][j] + mat2[i][j]
		}
	}
	return result
}

// MatSubtractFloat32 subtracts two matrices (2D slices) of type float32
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

// MatSubtract subtracts two matrices (2D slices) of type float64
func MatSubtract(mat1, mat2 [][]float64) [][]float64 {
	result := make([][]float64, len(mat1))
	for i := 0; i < len(mat1); i++ {
		result[i] = make([]float64, len(mat1[0]))
		for j := 0; j < len(mat1[i]); j++ {
			result[i][j] = mat1[i][j] - mat2[i][j]
		}
	}
	return result
}

// MatSumFloat32 sums all the values in a matrix (2D slice) of type float32
func MatSumFloat32(mat [][]float32) float32 {
	sum := float32(0.0)
	for i := 0; i < len(mat); i++ {
		for j := 0; j < len(mat[i]); j++ {
			sum += mat[i][j]
		}
	}
	return sum
}

// MatSum sums all the values in a matrix (2D slice) of type float64
func MatSum(mat [][]float64) float64 {
	sum := float64(0.0)
	for i := 0; i < len(mat); i++ {
		for j := 0; j < len(mat[i]); j++ {
			sum += mat[i][j]
		}
	}
	return sum
}

// Dot is the dot product of two vectors (1D slices) of type float64
func Dot(vec1, vec2 []float64) float64 {
	if len(vec1) != len(vec2) {
		log.Fatal("Invalid vector lengths. Vectors must have same length")
	}
	var product float64
	for i := range vec1 {
		product += vec1[i] * vec2[i]
	}
	return product
}

// MatMultFloat32 multiplies two matrices (2D slices) of type float32
func MatMultFloat32(mat1, mat2 [][]float32) [][]float32 {
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

// MatMult multiplies two matrices (2D slices) of type float64
func MatMult(mat1, mat2 [][]float64) [][]float64 {
	result := make([][]float64, len(mat1))
	for i := 0; i < len(mat1); i++ {
		result[i] = make([]float64, len(mat2[0]))
		for j := 0; j < len(mat2[0]); j++ {
			for k := 0; k < len(mat2); k++ {
				result[i][j] += mat1[i][k] * mat2[k][j]
			}
		}
	}
	return result
}

// ElemMultiplyFloat32 performs element-wise multiplication of two float32 vectors (slices) 
func ElemMultiplyFloat32(vec1, vec2 []float32) []float32 {
	result := make([]float32, len(vec1))
	for i := range vec1 {
		result[i] = vec1[i] * vec2[i]
	}
	return result
}

// ElemMultiply performs element-wise multiplication of two vectors (slices) 
func ElemMultiply(vec1, vec2 []float64) []float64 {
	result := make([]float64, len(vec1))
	for i := range vec1 {
		result[i] = vec1[i] * vec2[i]
	}
	return result
}

// ScalarMatMultFloat32 multiples each value in a matrix (2D slice) by the specified scalar
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

// ScalarMatMult multiples each value in a matrix (2D slice) by the specified scalar
func ScalarMatMult(scalar float64, mat [][]float64) [][]float64 {
	result := make([][]float64, len(mat))

	for i := 0; i < len(mat); i++ {
		result[i] = make([]float64, len(mat[i]))
		for j := 0; j < len(mat[i]); j++ {
			result[i][j] = scalar * mat[i][j]
		}
	}
	return result
}

// ScalarMatDivFloat32 divides each value in a matrix (2D slice) by the specified scalar
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

// ScalarMatDiv divides each value in a matrix (2D slice) by the specified scalar
func ScalarMatDiv(scalar float64, mat [][]float64) [][]float64 {
	result := make([][]float64, len(mat))

	for i := 0; i < len(mat); i++ {
		result[i] = make([]float64, len(mat[i]))
		for j := 0; j < len(mat[i]); j++ {
			result[i][j] = mat[i][j] / scalar
		}
	}
	return result
}

// MatPlusScalarFloat32 returns the resulting matrix (2D slice) after
// adding the specified scalar to each value in the matrix
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

// MatMinusScalarFloat32 returns the resulting matrix (2D slice) after
// subtracting the specified scalar from each value in the matrix
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

// TransposeFloat32 transposes a matrix (2D slice)
func TransposeFloat32(mat [][]float32) [][]float32 {
	result := make([][]float32, len(mat[0]))
	for i := 0; i < len(mat); i++ {
		for j := 0; j < len(mat[0]); j++ {
			result[j] = append(result[j], mat[i][j])
		}
	}
	return result
}

// Transpose transposes a matrix (2D slice)
func Transpose(mat [][]float64) [][]float64 {
	result := make([][]float64, len(mat[0]))
	for i := 0; i < len(mat); i++ {
		for j := 0; j < len(mat[0]); j++ {
			result[j] = append(result[j], mat[i][j])
		}
	}
	return result
}

// MeanFloat32 returns the average value of a slice of float32 values
func MeanFloat32(x []float32) float32 {
	sum := SumFloat32(x)
	return sum / float32(len(x))
}

// Mean returns the average value of a slice of float64 values
func Mean(x []float64) float64 {
	sum := Sum(x)
	return sum / float64(len(x))
}

// SumFloat32 returns the sum of the values in a slice of float32 numbers
func SumFloat32(x []float32) float32 {
	sum := float32(0)
	for _, val := range x {
		sum += val
	}
	return sum
}

// Sum returns the sum of the values in a slice of float32 numbers
func Sum(x []float64) float64 {
	sum := float64(0)
	for _, val := range x {
		sum += val
	}
	return sum
}

// MatMult1D multiplies two matrices that are passed in as one-dimensional slices
// and returns the resulting matrix. It uses goroutines that call a helper function
// to perform the calculations
func MatMult1D(matA, matB []float64, matARows, matBRows int) []float64 {
	matACols := len(matA) / matARows
	matBCols := len(matB) / matBRows
	matC := make([]float64, matARows*matBCols)
	nRoutines := 4
	nRowsPerRoutine := matARows/nRoutines
	specialCase := nRowsPerRoutine + (matARows%nRoutines)
	
	if matACols != matBRows {
		log.Fatal("Cannot multiply matrices. Incompatible matrix dimensions")
	}

	// create goroutines that perform sections of the matrix multiplication
	for i := 0; i < nRoutines; i++ {
		matCStartIndex := i*nRowsPerRoutine
		var matCEndIndex int
		// if number of rows in matA does not divide evenly by number of threads, last thread
		// takes on the remaining rows of matA in the multiplication  
		if (i == nRoutines-1 && nRowsPerRoutine != specialCase) {
            matCEndIndex = (i+1) * nRowsPerRoutine + (specialCase - nRowsPerRoutine);
        } else {
            matCEndIndex = (i+1) * nRowsPerRoutine;
		}
		go matMult1DHelper(matA, matB, matC, matCStartIndex, matCEndIndex, matACols, matBCols)
	}
	return matC
}

// matMult1DHelper performs matrix the matrix multiplication
func matMult1DHelper(matA, matB, matC []float64, matCStartIndex, matCEndIndex, matACols, matBCols int) {
	for i := matCStartIndex; i < matCEndIndex; i++ {
		for j := 0; j < matBCols; j++ {
			for k := 0; k < matACols; k++ {
				matC[i*matBCols+j] += matA[i*matACols+k]*matB[k*matBCols+j];
			}
		}
	}
}