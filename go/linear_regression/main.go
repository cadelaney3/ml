package main

import (
	"bufio"
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

//func cost()

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
		var y float32
		var e float32
		var meanSqErr float32
		var meanSqTemp float32
		y = a0 + a1*xTrain
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
	}
}
