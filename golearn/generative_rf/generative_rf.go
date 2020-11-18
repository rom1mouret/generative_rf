package generative_rf

import (
  "github.com/sjwhitworth/golearn/ensemble"
  "github.com/sjwhitworth/golearn/trees"
  "github.com/sjwhitworth/golearn/base"
  "math/rand"
  "math"
  "fmt"
)

type FeatGenerator struct {
  RF        *ensemble.RandomForest // contains Model *meta.BaggedModel
  ClassAttr base.Attribute         // we need to keep this around to run predict()
  Counters  []Counter
  Feat1Sum  map[string]float64     // sum of the values of each attribute
  Feat2Sum  map[string]float64     // sum of the squared values of each attribute
  Nrows     int                    // total number of rows
}

func (gen *FeatGenerator) Reinforce(data base.FixedDataGrid) *FeatGenerator {
  // count samples in each branch
  for i := 0; i < gen.RF.ForestSize; i++ {
    gen.Counters[i].count(gen.RF.Model.Models[i].(*trees.ID3DecisionTree), data)
  }
  return gen
}

func (gen *FeatGenerator) UpdateMoments(data base.FixedDataGrid) *FeatGenerator {
  // increment variables used to calculate mean and variance
  _, nRows := data.Size()
  for _, attr := range data.AllAttributes() {
    if attr.GetType() == base.Float64Type {
      spec := GetAttributeSpec(attr, data)
      var sum1 float64 = 0
      var sum2 float64 = 0
      for row := 0; row < nRows; row++ {
        val := base.UnpackBytesToFloat(data.Get(spec, row))
        sum1 += val
        sum2 += val * val
      }
      attrName :=  attr.(*base.FloatAttribute).Name
      gen.Feat1Sum[attrName] += sum1
      gen.Feat2Sum[attrName] += sum2
    }
  }
  gen.Nrows += nRows

  return gen
}

func (gen *FeatGenerator) Register(rf *ensemble.RandomForest, classAttr base.Attribute) *FeatGenerator {
  gen.RF = rf
  gen.Counters = make([]Counter, rf.ForestSize)
  gen.Feat1Sum = make(map[string]float64)
  gen.Feat2Sum = make(map[string]float64)
  gen.ClassAttr = classAttr
  return gen
}

func (gen FeatGenerator) StandardDeviation(attrName string) float64 {
  n := float64(gen.Nrows)
  sum := gen.Feat1Sum[attrName]
  besselVar := gen.Feat2Sum[attrName] / (n - 1) - (sum*sum) / ((n-1)*n)
  std := math.Sqrt(besselVar)
  return std
}

func (gen FeatGenerator) DefaultRandomFeatures(nSamples int) base.UpdatableDataGrid {
  ret := base.NewDenseInstances()

  // we add the attributes before calling Extend()
  specs := make(map[string]base.AttributeSpec)
  for attrName := range gen.Feat1Sum {
    attr := base.FloatAttribute{Name: attrName, Precision: 4}
    specs[attrName] = ret.AddAttribute(&attr) // adds one column
  }
  // adding the target column is weirdly required to call RandomForest.predict()
  ret.AddAttribute(gen.ClassAttr)
  err := ret.AddClassAttribute(gen.ClassAttr)
  if err != nil {
    panic(fmt.Sprintf("unable to mark attribute %s as class: %s",
          gen.ClassAttr.GetName(), err.Error()))
  }

  // make room for the values
  ret.Extend(nSamples) // allocate nSamples rows

  // generate random values one feature/attribute at a time
  for attrName, sum := range gen.Feat1Sum {
    std := gen.StandardDeviation(attrName)
    mean := sum / float64(gen.Nrows)
    spec := specs[attrName]
    for row := 0; row < nSamples; row++ {
        value := rand.NormFloat64() * std + mean
        ret.Set(spec, row, base.PackFloatToBytes(value))
    }
  }
  return ret
}

func (gen FeatGenerator) Generate(approxN int, minN int) (base.UpdatableDataGrid, error) {
  if approxN <= 0 {
    // auto mode
    approxN = int(math.Max(float64(minN), math.Sqrt(float64(gen.Nrows))))
  } else if minN > approxN {
    approxN = minN
  }

  // number of samples per tree
  nPerTree := approxN / gen.RF.ForestSize
  nSamples := gen.RF.ForestSize * nPerTree // actual number of generated samples

  // default values to cover features that are not set
  data := gen.DefaultRandomFeatures(nSamples)

  // calculate standard deviations to move around the decision boundary
  // at different amplitudes
  std := make(map[string]float64)
  for attrName, _ := range gen.Feat1Sum {
    std[attrName] = gen.StandardDeviation(attrName)
  }

  // generate the features
  rowNo := 0
  for i := 0; i < gen.RF.ForestSize; i++ {
    counter := gen.Counters[i]
    for n := 0; n < nPerTree; n++ {
      var nodeIndex NodeIndex
      row := make(map[base.FloatAttribute]float64)
      current := gen.RF.Model.Models[i].(*trees.ID3DecisionTree).Root
      for current.Children != nil && len(current.Children) == 2 {
        // splitting attribute and splitting value
        splitVal := current.SplitRule.SplitVal
        attr := *AssertFloat(current.SplitRule.SplitAttr)

        if val, already := row[attr]; already {
          // if the value has already been set, it's just regular branching
          if val <= splitVal {
            current = GetLeftChild(current)
            nodeIndex.MoveLeft()
          } else{
            current = GetRightChild(current)
            nodeIndex.MoveRight()
          }
        } else{
          // randomly choose a numerical value
          var generatedVal float64
          shift := math.Abs(rand.NormFloat64() * 0.05) * std[attr.Name]
          if rand.Float32() < counter.leftProbability(nodeIndex) {
            generatedVal = splitVal - shift
            current = GetLeftChild(current)
            nodeIndex.MoveLeft()
          } else {
            generatedVal = splitVal + shift
            current = GetRightChild(current)
            nodeIndex.MoveRight()
          }
          row[attr] = generatedVal
        }
      }
      // write the new row
      for attr, generatedVal := range row {
        spec := GetAttributeSpec(&attr, data)
        data.Set(spec, rowNo, base.PackFloatToBytes(generatedVal))
      }
      rowNo += 1
    }
  }

  // generate the targets
  predictions, err := gen.RF.Predict(data)
  if err != nil {
    return nil, err
  }
  err = CopyColumn(predictions.AllClassAttributes()[0], predictions, data)
  if err != nil {
    return nil, err
  }

  return data, nil
}
