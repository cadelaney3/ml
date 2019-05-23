package main

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func hypothesis(x float32, y float32) {

}

func matAddFloat32(mat1 [][]float32, mat2 [][]float32) ([][]float32, error) {
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

func matMultFloat32(mat1 [][]float32, mat2 [][]float32) ([][]float32, error) {
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

func transpose(mat [][]float32) [][]float32 {
	result := make([][]float32, len(mat[0]))
	for i := 0; i < len(mat); i++ {
		for j := 0; j < len(mat[0]); j++ {
			result[j] = append(result[j], mat[i][j])
		}
	}
	return result
}

func main() {

	// load data
	f, err := os.Open("../data/wine.data")
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

	// split data frame into train and test sets
	trainLen := int(float32(len(df)) * 0.8)
	//test_len := len(df) - train_len

	xTrainTemp := df[:trainLen]
	xTrain := make([][]float32, trainLen)
	yTrain := make([][]float32, trainLen)
	for i, x := range xTrainTemp {
		xTrain[i] = []float32{x[1] + x[2] + x[3] + x[4] + x[5] + x[6]}
		yTrain[i] = []float32{x[12]}
	}
	//fmt.Println(xTrain)

	//xTest := df[trainLen:]
	//fmt.Printf("len of x_train: %d\n", len(xTrain))
	//fmt.Printf("len of x_test: %d\n", len(xTest))

	n := len(xTrain)
	//alpha := 0.0001

	a0 := make([][]float32, n)
	a1 := make([][]float32, n)
	for index := range a0 {
		a0[index] = []float32{0}
		a1[index] = []float32{0}
	}
	//fmt.Println(a0)

	epochs := 0
	for epochs < 1000 {
		// var y float32
		// var e float32
		// var meanSqErr float32
		// var meanSqTemp float32
		//y = a0 + a1*xTrain
		a2 := transpose(a1)
		x, err := matMultFloat32(a2, xTrain)
		check(err)
		fmt.Println(x)
		/*
			e = y - yTrain
			for _, val := range e {
				meanSqTemp += val
			}
			errX := e * xTrain
			var sumErrX float32
			for _, val := range errX {
				sumErrX += val
			}
			meanSqErr = meanSqTemp * meanSqTemp
			meanSqErr = meanSqErr / n
			a0 = a0 - alpha*2*meanSqTemp/n
			a1 = a1 - alpha*2*sumErrX/n
			epochs += 1
			if epochs%10 == 0 {
				fmt.Println(meanSqErr)
			}
		*/
	}
}
