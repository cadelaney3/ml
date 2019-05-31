package main

import (
	"fmt"
	"time"

	"github.com/cadelaney3/ml/go/src/mlutils"
)

func main() {
	n := 16
	matA := make([]float64, n*n)
	matB := make([]float64, n*n)

	for i := range matA {
		matA[i] = 1
	}
	for i := range matB {
		matB[i] = 2
	}

	start := time.Now()
	matC := mlutils.MatMult1D(matA, matB, n, n)
	elapsed := time.Since(start)

	for i := 0; i < len(matC); i++ {
		if i%n == 0 {
			fmt.Println()
		}
		fmt.Printf("%.1f ", matC[i])
	}
	fmt.Printf("\nMatMult1D took %s\n", elapsed)
}
