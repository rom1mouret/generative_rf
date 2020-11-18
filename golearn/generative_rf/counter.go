package generative_rf

import (
  "github.com/sjwhitworth/golearn/trees"
  "github.com/sjwhitworth/golearn/base"
)

// structure to calculate probabilities by counting samples
type Counter struct {
  Count  []int
}

func (counter *Counter) count(tree *trees.ID3DecisionTree, data base.FixedDataGrid) {
  if counter.Count == nil {
    counter.Count = make([]int, maxNodeIndex(tree) + 1)
  }

  // same as execution flow as for prediction:
  attrSpecs := base.ResolveAttributes(data, data.AllAttributes())
  data.MapOverRows(attrSpecs, func(row [][]byte, rowNo int) (bool, error) {
    current := tree.Root
    var nodeIndex NodeIndex
    for current.Children != nil && len(current.Children) == 2 {
      splitVal := current.SplitRule.SplitVal
      attr := AssertFloat(current.SplitRule.SplitAttr)
      spec := GetAttributeSpec(attr, data)
      classVal := base.UnpackBytesToFloat(data.Get(spec, rowNo))
      if classVal <= splitVal {
        current = GetLeftChild(current)
        counter.Count[nodeIndex.MoveLeft()]++
      } else {
        current = GetRightChild(current)
        counter.Count[nodeIndex.MoveRight()]++
      }
    }
    return true, nil
  })

}

func (counter Counter) leftProbability(nodeIndex NodeIndex) float32 {
  left := float32(counter.Count[nodeIndex.Left()])
  right := float32(counter.Count[nodeIndex.Right()])
  return (1.0 * left) / (left + right)
}
