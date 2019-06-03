package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strconv"
	"unicode"

	"github.com/cadelaney3/ml/go/src/mlutils"
)

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func isNumber(s string) bool {
	for _, val := range s {
		if unicode.IsLetter(val) {
			return false
		}
		if string(val) == string("?") {
			return false
		}
	}
	return true
}

func processCSV(path string) [][]float64 {
	var df [][]float64

	f, err := os.Open(path)
	check(err)
	defer f.Close()

	r := csv.NewReader(f)
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		temp := make([]float64, len(record))
		for j, val := range record {
			if isNumber(val) {
				t, err := strconv.ParseFloat(val, 64)
				temp[j] = float64(t)
				check(err)
			} else {
				if val == "?" {
					temp[j] = float64(1)
				}
			}
		}
		df = append(df, temp)
	}
	return df
}

func sigmoid(t [][]float64) [][]float64 {
	sig := make([][]float64, len(t))
	for i, x := range t {
		sig[i] = make([]float64, len(x))
		for j, val := range x {
			sig[i][j] = 1 / (1 + math.Exp(-val))
		}
	}
	return sig
}

func h(theta, x [][]float64) [][]float64 {
	return mlutils.MatMult(x, theta)
}

func hypothesis(theta, x [][]float64) [][]float64 {
	return sigmoid(h(theta, x))
}

func cost(theta, x, y [][]float64) float64 {
	m := float64(len(x))
	hyp := hypothesis(theta, x)
	insideSumLeft := mlutils.ElemMatMult(y, mlutils.Log2D(hyp))
	insideSumRight := mlutils.ElemMatMult(mlutils.SubMatFromScalar(float64(1), y), mlutils.Log2D(mlutils.SubMatFromScalar(1, hyp)))
	return float64(-(1 / m)) * mlutils.MatSum(mlutils.MatAdd(insideSumLeft, insideSumRight))
}

func gradient(theta, x, y [][]float64) [][]float64 {
	m := float64(len(x))
	xT := mlutils.Transpose(x)
	hyp := hypothesis(theta, x)
	residuals := mlutils.MatSubtract(hyp, y)
	insideSum := mlutils.MatMult(xT, residuals)
	return mlutils.ScalarMatMult(float64(1/m), insideSum)
}

func updateTheta(theta, x, y [][]float64, eta float64) [][]float64 {
	grad := gradient(theta, x, y)
	grad = mlutils.ScalarMatMult(eta, grad)
	theta = mlutils.MatSubtract(theta, grad)
	return theta
}

func fit(theta, x, y [][]float64, eta float64, iterations int) ([][]float64, []float64) {
	var costVec []float64

	for i := 0; i < iterations; i++ {
		theta = updateTheta(theta, x, y, eta)
		costIter := cost(theta, x, y)
		costVec = append(costVec, costIter)
		if i%1000 == 0 {
			fmt.Printf("Iter: %d cost: %f\n", i, costIter)
		}
	}
	return theta, costVec
}

func predict(theta, x [][]float64) [][]float64 {
	predictions := mlutils.MatMult(x, theta)
	for i, xi := range predictions {
		for j := range xi {
			if predictions[i][j] < 0.5 {
				predictions[i][j] = 0
			} else {
				predictions[i][j] = 1
			}
		}
	}
	return predictions
}

func accuracy(predictions, actual [][]float64) float64 {
	diff := mlutils.MatSubtract(predictions, actual)
	count := 0
	for i, xi := range diff {
		for j := range xi {
			if diff[i][j] != 0 {
				count++
			}
		}
	}
	return float64(1) - (float64(count) / float64(len(diff)))
}

func main() {
	path := "../../../data/breast-cancer-wisconsin.data"
	df := processCSV(path)

	x := make([][]float64, len(df))
	y := make([][]float64, len(df))

	xCols := len(df[0]) - 1
	for i, v := range df {
		x[i] = v[1:xCols]
		y[i] = v[xCols:]
	}

	for i, xi := range y {
		for j := range xi {
			if y[i][j] == 2 {
				y[i][j] = 0
			} else {
				y[i][j] = 1
			}
		}
	}

	for i, xi := range x {
		x[i] = append([]float64{1}, xi...)
	}

	theta := mlutils.ZerosMat(xCols, 1)

	fmt.Printf("Shape of x: %s\n", mlutils.Shape(x))
	fmt.Printf("Shape of y: %s\n", mlutils.Shape(y))
	fmt.Printf("Shape of theta: %s\n", mlutils.Shape(theta))

	theta, _ = fit(theta, x, y, 0.0001, 5000)
	yPred := predict(theta, x)

	fmt.Println(accuracy(yPred, y))
}
