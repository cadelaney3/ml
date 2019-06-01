package mlutils

import (
	"math"
)

// StandardDevFloat32 returns the standard deviation of a slice of float32 values
func StandardDevFloat32(x []float32) float32 {
	n := float32(len(x))
	xBar := MeanFloat32(x)
	numerator := make([]float32, len(x))

	for i, val := range x {
		numerator[i] = (val - xBar) * (val - xBar)
	}
	return float32(math.Sqrt(float64(SumFloat32(numerator) / n)))
}

// StandardDev returns the standard deviation of a slice of float64 values
func StandardDev(x []float64) float64 {
	n := float64(len(x))
	xBar := Mean(x)
	numerator := make([]float64, len(x))

	for i, val := range x {
		numerator[i] = (val - xBar) * (val - xBar)
	}
	return math.Sqrt(float64(Sum(numerator) / n))
}

// R2Float32 takes in two slices and calculates the r2 score (coefficient of determination)
func R2Float32(x, y []float32) float32 {
	n := float32(len(x))
	meanX := MeanFloat32(x)
	meanY := MeanFloat32(y)
	deltaX := StandardDevFloat32(x)
	deltaY := StandardDevFloat32(y)
	summation := make([]float32, len(x))
	for i := range x {
		summation[i] = (x[i] - meanX) * (y[i] - meanY)
	}
	r := (float32(1) / n) * SumFloat32(summation) / (deltaX * deltaY)
	return r * r
}

// R2 takes in two slices and calculates the r2 score (coefficient of determination)
func R2(x, y []float64) float64 {
	n := float64(len(x))
	meanX := Mean(x)
	meanY := Mean(y)
	deltaX := StandardDev(x)
	deltaY := StandardDev(y)
	summation := make([]float64, len(x))
	for i := range x {
		summation[i] = (x[i] - meanX) * (y[i] - meanY)
	}
	r := (float64(1) / n) * Sum(summation) / (deltaX * deltaY)
	return r * r
}