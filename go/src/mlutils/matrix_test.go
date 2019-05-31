package mlutils

import (
	"math/rand"
	"testing"
	"time"
)

func BenchmarkMatMult1D(b *testing.B) {
	s1 := rand.NewSource(time.Now().UnixNano())
	r1 := rand.New(s1)
	n := 64
	matA := make([]float64, n*n)
	matB := make([]float64, n*n)

	for i := range matA {
		matA[i] = r1.Float64()
	}
	for i := range matB {
		matB[i] = r1.Float64()
	}
	for i := 0; i < b.N; i++ {
		MatMult1D(matA, matB, n, n)
	}
}