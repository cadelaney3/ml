package matrix

import "errors"

func MatAddFloat32(mat1 [][]float32, mat2 [][]float32) ([][]float32, error) {
	if len(mat1) != len(mat2) && len(mat1[0]) != len(mat2[0]) {
		return nil, errors.New("invalid matrix dimensions for matrix addition")
	}

	result := make([][]float32, len(mat1))
	for i := 0; i < len(mat1); i++ {
		result[i] = make([]float32, len(mat1[0]))
		for j := 0; j < len(mat1); j++ {
			result[i][j] = mat1[i][j] + mat2[i][j]
		}
	}
	return result, nil
}

func MatSubtractFloat32(mat1 [][]float32, mat2 [][]float32) ([][]float32, error) {
	if len(mat1) != len(mat2) && len(mat1[0]) != len(mat2[0]) {
		return nil, errors.New("invalid matrix dimensions for matrix subtraction")
	}

	result := make([][]float32, len(mat1))
	for i := 0; i < len(mat1); i++ {
		result[i] = make([]float32, len(mat1[0]))
		for j := 0; j < len(mat1); j++ {
			result[i][j] = mat1[i][j] - mat2[i][j]
		}
	}
	return result, nil
}

func MatSumFloat32(slice [][]float32) float32 {
	sum := float32(0.0)
	for i := 0; i < len(slice); i++ {
		for j := 0; j < len(slice[i]); j++ {
			sum += slice[i][j]
		}
	}
	return sum
}

func MatMultFloat32(mat1 [][]float32, mat2 [][]float32) ([][]float32, error) {
	if len(mat1[0]) != len(mat2) {
		return nil, errors.New("invalid matrix dimensions for matrix multiplication")
	}
	//fmt.Printf("len mat1: %d; len mat2: %d\n", len(mat1), len(mat2))
	result := make([][]float32, len(mat1))
	for i := 0; i < len(mat1); i++ {
		result[i] = make([]float32, len(mat2[0]))
		for j := 0; j < len(mat1); j++ {
			for k := 0; k < len(mat1); k++ {
				result[i][j] += mat1[i][k] * mat2[k][j]
			}
		}
	}
	return result, nil
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
