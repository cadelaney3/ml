package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
	"unicode"

	"github.com/cadelaney3/ml/go/src/mlutils"
)

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func readCSV(path string) [][]float32 {
	// load data
	//f, err := os.Open("../data/wine.data")
	f, err := os.Open(path)
	check(err)
	defer f.Close()

	// create 2D slice dataframe
	var df [][]float32
	// scan thru each line of file
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		// is the each line as one string
		s := scanner.Text()
		// split s, but each value in array is still a string
		splitS := strings.Split(s, ",")
		fmt.Println(splitS)
		var line []float32
		// convert each value in split_s to float32
		for _, x := range splitS {
			temp, err := strconv.ParseFloat(x, 32)
			check(err)
			temp2 := float32(temp)
			line = append(line, temp2)
		}
		df = append(df, line)
	}
	//fmt.Println(df)

	if err := scanner.Err(); err != nil {
		panic(err)
	}
	return df
}

func isNumber(s string) bool {
	for _, val := range s {
		if unicode.IsLetter(val) {
			return false
		}
	}
	return true
}

func processCSV(path string) [][]float32 {
	var df [][]float32

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
		temp := make([]float32, len(record))
		fmt.Println(record)
		for j, val := range record {
			if isNumber(val) {
				t, err := strconv.ParseFloat(val, 32)
				temp[j] = float32(t)
				check(err)
			}

		}
		df = append(df, temp)
	}
	return df
}

func main() {

	dfTrain := processCSV("../data/random_linear_train.csv")
	dfTest := processCSV("../data/random_linear_test.csv")

	xTrain := make([][]float32, len(dfTrain))
	xTest := make([][]float32, len(dfTest))
	yTrain := make([][]float32, len(dfTrain))
	yTest := make([][]float32, len(dfTest))

	for i, x := range dfTrain {
		xTrain[i] = []float32{x[0]}
		yTrain[i] = []float32{x[1]}
	}
	for i, x := range dfTest {
		xTest[i] = []float32{x[0]}
		yTest[i] = []float32{x[1]}
	}

	n := len(xTrain)
	alpha := float32(0.0001)

	a0 := make([][]float32, n)
	a1 := make([][]float32, n)
	for index := range a0 {
		a0[index] = []float32{0}
		a1[index] = []float32{0}
	}

	epochs := 0
	for epochs < 1000 {

		x, err := mlutils.MatMultFloat32(a1, mlutils.Transpose(xTrain))
		check(err)

		y, err := mlutils.MatAddFloat32(a0, x)
		check(err)

		e, err := mlutils.MatSubtractFloat32(y, yTrain)
		check(err)

		e2, err := mlutils.MatMultFloat32(e, mlutils.Transpose(e))
		check(err)

		meanSqErr := mlutils.MatSumFloat32(mlutils.Transpose(e2))

		meanSqErr = meanSqErr / float32(n)

		a0 = mlutils.MatMinusScalarFloat32(a0, (alpha * float32(2) * mlutils.MatSumFloat32(e) / float32(n)))

		eXxTrain, err := mlutils.MatMultFloat32(e, mlutils.Transpose(xTrain))
		check(err)

		a1 = mlutils.MatMinusScalarFloat32(a1, (alpha * float32(2) * mlutils.MatSumFloat32(eXxTrain) / float32(n)))

		epochs++
		if epochs%10 == 0 {
			fmt.Println(meanSqErr)
		}

	}

	a1XxTest, err := mlutils.MatMultFloat32(a1, mlutils.Transpose(xTest))
	check(err)
	fmt.Println(a1XxTest)

	yPrediction, err := mlutils.MatAddFloat32(a0, a1XxTest)
	check(err)
	fmt.Println(a0)
	fmt.Println(yPrediction)

	yTestTranspose := mlutils.Transpose(yTest)
	//fmt.Println(yTestTranspose)

	yPredictionTranspose := mlutils.Transpose(yPrediction)
	//fmt.Println(yPredictionTranspose)

	fmt.Printf("R2 Score: %f\n", mlutils.R2(yTestTranspose[0], yPredictionTranspose[0]))

}
