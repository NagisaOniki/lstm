//RNNおよびLSTMを用いて英文の次文字予測変換モデルを作成し精度を比較する

object RNN{

  import breeze.linalg._

  class Affine(val input_size:Int, val output_size:Int, val lr:Double){

    var W = DenseMatrix.zeros[Double](output_size, input_size).map(_ => rand.nextGaussian * 0.01)
    var b = DenseVector.zeros[Double](output_size)
    var x1 = List[DenseVector[Double]]()
    var dW = DenseMatrix.zeros[Double](W.rows, W.cols)
    var db = DenseVector.zeros[Double](b.size)

    def forward(x:DenseVector[Double]) = {
      x1 ::= x
      W * x + b
    }

    def backward(d:DenseVector[Double]) = {
      val dx = W.t * d
      dW += d * x1.head.t
      db += d
      x1 = x1.tail
      dx
    }

    def update() {
      W -= lr * dW
      b -= lr * db
      dW = DenseMatrix.zeros[Double](W.rows, W.cols)
      db = DenseVector.zeros[Double](b.size)
    }
  }

  class Sigmoid(){
    var ys = List[DenseVector[Double]]()

    def sigmoid(x:Double) = 1 / (1 + math.exp(-x))

    def forward(x:DenseVector[Double]) = {
      ys ::= x.map(sigmoid)
      ys.head
    }

    def backward(d:DenseVector[Double]) = {
      val y = ys.head
      ys = ys.tail
      d *:* y *:* (1d - y)
    }
  }

  val n=27
  val m=27
  val rand = new util.Random(0)
  var target = "abcdefghijklmnopqrstuvwxyz "
  val lr = 0.1 //学習率

  var bx = DenseVector.zeros[Double](m)
  var Wh = DenseMatrix.zeros[Double](m,m).map(_=>rand.nextGaussian*0.01)
  var Wx = DenseMatrix.zeros[Double](m,n).map(_=>rand.nextGaussian*0.01)
  var dr = DenseVector.zeros[Double](m)
  var dx = DenseVector.zeros[Double](m)
  var dbx = DenseVector.zeros[Double](m)
  var dWh = DenseMatrix.zeros[Double](m,m)
  var dWx = DenseMatrix.zeros[Double](m,n)
  var xs = List[DenseVector[Double]](DenseVector.zeros[Double](n))
  var hs = List[DenseVector[Double]](DenseVector.zeros[Double](m)) //出力のリスト

  val af = new Affine(m,n,lr)
  val sig = new Sigmoid()

  //順方向の計算
  def forwards(x:DenseVector[Double])={
    val hprev = hs.head
    val h = (Wx * x + Wh * hprev + bx).map(tanh)
    xs = x :: xs
    hs = h :: hs
    h
  }

  //逆方向の計算
  def backwards(dh:DenseVector[Double])={ 
    val du = (dh + dr) *:* (1d - hs.head *:* hs.head)
    
    //dr,dxを更新
    dr = Wh.t * du
    dx = Wx.t * du

    dbx += du
    dWh += du * hs.tail.head.t
    dWx += du * xs.head.t

    xs = xs.tail
    hs = hs.tail

    dx
  }
  
  def update()={
    bx -= lr * dbx
    Wx -= lr * dWx
    Wh -= lr * dWh

    dbx = DenseVector.zeros[Double](m)
    dWx = DenseMatrix.zeros[Double](m,m)
    dWh = DenseMatrix.zeros[Double](m,m)

    dr = DenseVector.zeros[Double](m)
  }
  
  def tanh(x:Double)={
    (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
  }

  def sigmoid(u:Double)={
    1/(1+math.exp(-u))
  }

  //学習
  def learning(){
    val input = load("/home/share/text8")

    for(i<-0 until 300){
      var correct = 0
      var err = 0d
      var output = ""
      var answer = ""
      for(j<-0 until (input.size-5)/2){

        val y1 = sig.forward(af.forward(forwards(input(j))))
        val y2 = sig.forward(af.forward(forwards(input(j+1))))
        val y3 = sig.forward(af.forward(forwards(input(j+2))))
        val y4 = sig.forward(af.forward(forwards(input(j+3))))
        val y5 = sig.forward(af.forward(forwards(input(j+4))))
  /*      val y6 = sig.forward(af.forward(forwards(input(j+5))))
        val y7 = sig.forward(af.forward(forwards(input(j+6))))
        val y8 = sig.forward(af.forward(forwards(input(j+7))))
        val y9 = sig.forward(af.forward(forwards(input(j+8))))
        val y10 = sig.forward(af.forward(forwards(input(j+9))))
        val y11 = sig.forward(af.forward(forwards(input(j+10))))
        val y12 = sig.forward(af.forward(forwards(input(j+11))))
        val y13 = sig.forward(af.forward(forwards(input(j+12))))
        val y14 = sig.forward(af.forward(forwards(input(j+13))))
        val y15 = sig.forward(af.forward(forwards(input(j+14))))
*/
        if(argmax(y5) == argmax(input(j+5))){
          correct += 1
        }
        err += - math.log(y5(argmax(input(j+5))))

        output += target(argmax(y5))
        answer += target(argmax(input(j+5)))
/*
        backwards(af.backward(sig.backward(y15-input(j+15))))
        backwards(af.backward(sig.backward(y14-input(j+14))))
        backwards(af.backward(sig.backward(y13-input(j+13))))
        backwards(af.backward(sig.backward(y12-input(j+12))))
        backwards(af.backward(sig.backward(y11-input(j+11))))
        backwards(af.backward(sig.backward(y10-input(j+10))))
        backwards(af.backward(sig.backward(y9-input(j+9))))
        backwards(af.backward(sig.backward(y8-input(j+8))))
        backwards(af.backward(sig.backward(y7-input(j+7))))
        backwards(af.backward(sig.backward(y6-input(j+6))))*/
        backwards(af.backward(sig.backward(y5-input(j+5))))
        backwards(af.backward(sig.backward(y4-input(j+4))))
        backwards(af.backward(sig.backward(y3-input(j+3))))
        backwards(af.backward(sig.backward(y2-input(j+2))))
        backwards(af.backward(sig.backward(y1-input(j+1))))

        update()
        af.update()
      }

      //println(((correct/((input.size-10)/2).toDouble)*100).toInt)
      println(err/((input.size-5)/2))
      //println("<output>")
      //println(output)
      //println("<answer>")
      //println(answer)
      //println("")
    }

    //test
    var correct2 = 0
    var err2 = 0d
    var output2 = ""
    var answer2 = ""
    for(j <- (input.size-5)/2 until input.size-5){

      val y1 = sig.forward(af.forward(forwards(input(j))))
      val y2 = sig.forward(af.forward(forwards(input(j+1))))
      val y3 = sig.forward(af.forward(forwards(input(j+2))))
      val y4 = sig.forward(af.forward(forwards(input(j+3))))
      val y5 = sig.forward(af.forward(forwards(input(j+4))))
     /* val y6 = sig.forward(af.forward(forwards(input(j+5))))
      val y7 = sig.forward(af.forward(forwards(input(j+6))))
      val y8 = sig.forward(af.forward(forwards(input(j+7))))
      val y9 = sig.forward(af.forward(forwards(input(j+8))))
      val y10 = sig.forward(af.forward(forwards(input(j+9))))
      val y11 = sig.forward(af.forward(forwards(input(j+10))))
      val y12 = sig.forward(af.forward(forwards(input(j+11))))
      val y13 = sig.forward(af.forward(forwards(input(j+12))))
      val y14 = sig.forward(af.forward(forwards(input(j+13))))
      val y15 = sig.forward(af.forward(forwards(input(j+14))))
   */
      if(argmax(y5) == argmax(input(j+5))){
        correct2 += 1
      }
      err2 += - math.log(y5(argmax(input(j+5))))

      output2 += target(argmax(y5))
      answer2 += target(argmax(input(j+5)))
/*
      backwards(af.backward(sig.backward(y15-input(j+15))))
      backwards(af.backward(sig.backward(y14-input(j+14))))
      backwards(af.backward(sig.backward(y13-input(j+13))))
      backwards(af.backward(sig.backward(y12-input(j+12))))
      backwards(af.backward(sig.backward(y11-input(j+11))))
      backwards(af.backward(sig.backward(y10-input(j+10))))
      backwards(af.backward(sig.backward(y9-input(j+9))))
      backwards(af.backward(sig.backward(y8-input(j+8))))
      backwards(af.backward(sig.backward(y7-input(j+7))))
      backwards(af.backward(sig.backward(y6-input(j+6))))*/
      backwards(af.backward(sig.backward(y5-input(j+5))))
      backwards(af.backward(sig.backward(y4-input(j+4))))
      backwards(af.backward(sig.backward(y3-input(j+3))))
      backwards(af.backward(sig.backward(y2-input(j+2))))
      backwards(af.backward(sig.backward(y1-input(j+1))))

      update()
      af.update()
    }
    println("------test-------")
    println("correct:" + ((correct2/((input.size-5)/2).toDouble)*100).toInt + "%")
    println("err:" + err2/((input.size-5)/2))
    println("<output>")
    println(output2)
    println("<answer>")
    println(answer2)
    println("")
  }

  def load(fn:String)={
    val file = io.Source.fromFile(fn).getLines.toArray.head.split(" ").take(500)
    val ws = file.mkString(" ")
    println("ws:" + ws)
    ws.map(conv).toArray
  }

  def conv(c:Char)={
    val x = DenseVector.zeros[Double](m)
    x(target.indexOf(c)) = 1d
    x
  }

}
