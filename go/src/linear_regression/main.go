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
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

type xy struct {
	x []float64
	y []float64
}

func (d xy) Len() int {
	return len(d.x)
}

func (d xy) XY(i int) (x, y float64) {
	x = d.x[i]
	y = d.y[i]
	return
}

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

func bestFitSlopeAndIntercept(x, y [][]float32) (float32, float32) {
	var xT [][]float32
	var yT [][]float32
	if len(x) > 1 {
		xT = mlutils.Transpose(x)
	} else {
		xT = x
		x = mlutils.Transpose(x)
	}
	if len(y) > 1 {
		yT = mlutils.Transpose(y)
	} else {
		yT = y
		y = mlutils.Transpose(y)
	}

	xXy := mlutils.Multiply(xT[0], yT[0])
	xXx := mlutils.Multiply(xT[0], xT[0])
	meanX := mlutils.MeanFloat32(xT[0])
	meanY := mlutils.MeanFloat32(yT[0])

	m := ((meanX * meanY) - mlutils.MeanFloat32(xXy)) /
		((meanX * meanX) - mlutils.MeanFloat32(xXx))

	b := meanY - m*meanX
	return m, b
}

func coefficientOfDetermination(yTest, yLine [][]float32) float32 {
	var yTestT [][]float32
	if len(yTest) > 1 {
		yTestT = mlutils.Transpose(yTest)
	} else {
		yTestT = yTest
		yTest = mlutils.Transpose(yTest)
	}

	yMeanLine := make([][]float32, 1)
	yMeanLine[0] = make([]float32, len(yTestT[0]))
	yTestMean := mlutils.MeanFloat32(yTestT[0])
	for i := range yTest[0] {
		yMeanLine[0][i] = yTestMean
	}

	temp := mlutils.MatSubtractFloat32(yLine, yTest)
	tempSq := mlutils.MatMultFloat32(mlutils.Transpose(temp), temp)
	tempSq = mlutils.Transpose(tempSq)
	sqErrRegress := mlutils.SumFloat32(tempSq[0])

	temp2 := mlutils.MatSubtractFloat32(yMeanLine, yTestT)
	temp2Sq := mlutils.MatMultFloat32(mlutils.Transpose(temp2), temp2)
	temp2Sq = mlutils.Transpose(temp2Sq)
	sqErrYMean := mlutils.SumFloat32(temp2Sq[0])

	rSquared := 1 - (sqErrRegress / sqErrYMean)
	return rSquared
}

func main() {

	dfTrain := processCSV("../../../data/random_linear_train.csv")
	dfTest := processCSV("../../../data/random_linear_test.csv")

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

	xTrain = xTrain[1:]
	yTrain = yTrain[1:]
	xTest = xTest[1:]
	yTest = yTest[1:]

	m, b := bestFitSlopeAndIntercept(xTrain, yTrain)
	regressionLine := make([][]float32, len(xTest))
	for i, x := range xTest {
		regressionLine[i] = make([]float32, 1)
		regressionLine[i][0] = (m * x[0]) + b
	}
	rSquared := coefficientOfDetermination(yTest, regressionLine)
	fmt.Printf("R2 score: %f\n", rSquared)

	line := plotter.NewFunction(func(x float64) float64 { return float64(m)*x + float64(b) })

	p, err := plot.New()
	if err != nil {
		log.Panic(err)
	}

	plotter.DefaultLineStyle.Width = vg.Points(1)
	plotter.DefaultGlyphStyle.Radius = vg.Points(2)

	x := make([]float64, len(xTrain))
	y := make([]float64, len(yTrain))
	xT := mlutils.Transpose(xTrain)[0]
	yT := mlutils.Transpose(yTrain)[0]

	for i := range xT {
		x[i] = float64(xT[i])
		y[i] = float64(yT[i])
	}

	data := xy{
		x: x,
		y: y,
	}

	scatter, err := plotter.NewScatter(data)
	if err != nil {
		log.Panic(err)
	}
	p.Add(scatter, line)

	err = p.Save(200, 200, "scatter.png")
	if err != nil {
		log.Panic(err)
	}
}
