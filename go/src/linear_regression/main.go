package main

import (
	"encoding/csv"
	"fmt"
	"image/color"
	"io"
	"log"
	"os"
	"strconv"
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

	p, err := plot.New()
	if err != nil {
		log.Panic(err)
	}
	p.Title.Text = "Best Fit Line"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	plotter.DefaultLineStyle.Width = vg.Points(1)
	plotter.DefaultGlyphStyle.Radius = vg.Points(2)

	x := make([]float64, len(xTest))
	y := make([]float64, len(yTest))
	xT := mlutils.Transpose(xTest)[0]
	yT := mlutils.Transpose(yTest)[0]
	regLine := make([]float64, len(regressionLine))

	for i := range xT {
		x[i] = float64(xT[i])
		y[i] = float64(yT[i])
	}
	for i := range regressionLine {
		regLine[i] = float64(regressionLine[i][0])
	}

	data := xy{
		x: x,
		y: y,
	}
	lineData := xy{
		x: x,
		y: regLine,
	}

	scatter, err := plotter.NewScatter(data)
	if err != nil {
		log.Panic(err)
	}
	l, err := plotter.NewLine(lineData)
	if err != nil {
		log.Panic(err)
	}
	l.LineStyle.Width = vg.Points(1)
	l.LineStyle.Color = color.RGBA{R: 255, A: 255}
	p.Add(scatter, l)

	err = p.Save(200, 200, "scatter.png")
	if err != nil {
		log.Panic(err)
	}
}
