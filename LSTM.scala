//RNNおよびLSTMを用いて英文の次文字予測変換モデルを作成し精度を比較する

import breeze.linalg._
import CLASS._

object LSTM{

  val n=27 //入力サイズ
  val m=27 //出力サイズ
  val rand = new util.Random(0)
  var target = "abcdefghijklmnopqrstuvwxyz "
  val lr = 0.1 //学習率

/*
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

  var bi = DenseVector.zeros[Double](m)
  var bf = DenseVector.zeros[Double](m)
  var bo = DenseVector.zeros[Double](m)
  var bh_ = DenseVector.zeros[Double](m)
  var Wir = DenseMatrix.zeros[Double](m,m).map(_=>rand.nextGaussian*0.01)
  var Wfr = DenseMatrix.zeros[Double](m,m).map(_=>rand.nextGaussian*0.01)
  var Wor = DenseMatrix.zeros[Double](m,m).map(_=>rand.nextGaussian*0.01)
  var Wh_r = DenseMatrix.zeros[Double](m,m).map(_=>rand.nextGaussian*0.01)
  var Wix = DenseMatrix.zeros[Double](m,n).map(_=>rand.nextGaussian*0.01)
  var Wfx = DenseMatrix.zeros[Double](m,n).map(_=>rand.nextGaussian*0.01)
  var Wox = DenseMatrix.zeros[Double](m,n).map(_=>rand.nextGaussian*0.01)
  var Wh_x = DenseMatrix.zeros[Double](m,n).map(_=>rand.nextGaussian*0.01)

  var dC = DenseVector.zeros[Double](m)
  var dr = DenseVector.zeros[Double](m)
  var dx = DenseVector.zeros[Double](m)

  var dbi = DenseVector.zeros[Double](m)
  var dbf = DenseVector.zeros[Double](m)
  var dbo = DenseVector.zeros[Double](m)
  var dbh_ = DenseVector.zeros[Double](m)
  var dWir = DenseMatrix.zeros[Double](m,m)
  var dWfr = DenseMatrix.zeros[Double](m,m)
  var dWor = DenseMatrix.zeros[Double](m,m)
  var dWh_r = DenseMatrix.zeros[Double](m,m)
  var dWix = DenseMatrix.zeros[Double](m,n)
  var dWfx = DenseMatrix.zeros[Double](m,n)
  var dWox = DenseMatrix.zeros[Double](m,n)
  var dWh_x = DenseMatrix.zeros[Double](m,n)

  var is = List[DenseVector[Double]](DenseVector.zeros[Double](n))
  var h_s =List[DenseVector[Double]](DenseVector.zeros[Double](n))
  var fs = List[DenseVector[Double]](DenseVector.zeros[Double](n))
  var os = List[DenseVector[Double]](DenseVector.zeros[Double](n))
  var xs = List[DenseVector[Double]](DenseVector.zeros[Double](n))
  var cs = List[DenseVector[Double]](DenseVector.zeros[Double](m)) //記憶のリスト
  var hs = List[DenseVector[Double]](DenseVector.zeros[Double](m)) //出力のリスト
 
 def forwards(x:DenseVector[Double])={
 val hprev = hs.head
 val Cprev = cs.head
 val i = (Wix * x + Wir * hprev + bi).map(sigmoid)
 val h_ = (Wh_x * x + Wh_r * hprev + bh_).map(tanh)
 val f = (Wfx * x + Wfr * hprev + bf).map(sigmoid)
    val o = (Wox * x + Wor * hprev + bo).map(sigmoid)
    val C = i *:* h_ + f *:* Cprev
    val h = o * C.map(tanh)
    is = i :: is
    h_s = h_ :: h_s
    fs = f :: fs
    os = o :: os
    xs = x :: xs
    cs = C :: cs
    hs = h :: hs
    h
  }
  def backwards(dh:DenseVector[Double])={ //引数はdはdhのこと
    val Z = cs.head.map(tanh)
    val A = os.head *:* (dh + dr) *:* (1d - Z *:* Z)   
    val dbit = ( dC + A ) *:* h_s.head * is.head * ( 1d - is.head )
    val dbft = ( dC + A ) * cs.tail.head * fs.head * ( 1d - fs.head )
    val dbot = ( dh + dr ) * cs.head.map(tanh) * os.head * ( 1d - os.head )
    val dbh_t = ( dC + A ) * is.head * ( 1d - h_s.head * h_s.head )
    //dC,dr,dxを更新
    dC = ( dC + A ) *:* fs.head
    dr = Wor.t * dbot + Wfr.t * dbft +  Wh_r.t * dbh_t + Wir.t * dbit
    dx = Wox.t * dbot + Wfx.t * dbft +  Wix.t * dbh_t + Wix.t * dbit
    dbi += dbit
    dbf += dbft
    dbo += dbot
    dbh_ += dbh_t
    val dWirt = dbit * hs.tail.head.t
    val dWfrt = dbft * hs.tail.head.t
    val dWort = dbot * hs.tail.head.t
    val dWh_rt = dbh_t * hs.tail.head.t
    dWir += dWirt
    dWfr += dWfrt
    dWor += dWort
    dWh_r += dWh_rt
    val dWixt = dbit * xs.head.t
    val dWfxt = dbft * xs.head.t
    val dWoxt = dbot * xs.head.t
    val dWh_xt = dbh_t * xs.head.t
    dWix += dWixt
    dWfx += dWfxt
    dWox += dWoxt
    dWh_x += dWh_xt
    is = is.tail
    h_s = h_s.tail
    fs = fs.tail
    os = os.tail
    xs = xs.tail
    cs = cs.tail
    hs = hs.tail
    dx
  }
  def update()={
    bi -= lr * dbi
    bf -= lr * dbf
    bo -= lr * dbo
    bh_ -= lr * dbh_
    Wir -= lr * dWir
    Wfr -= lr * dWfr
    Wor -= lr * dWor
    Wh_r -= lr * dWh_r
    Wix -= lr * dWix
    Wfx -= lr * dWfx
    Wox -= lr * dWox
    Wh_x -= lr * dWh_x
    dbi = DenseVector.zeros[Double](m)
    dbf = DenseVector.zeros[Double](m)
    dbo = DenseVector.zeros[Double](m)
    dbh_ = DenseVector.zeros[Double](m)
    dWir = DenseMatrix.zeros[Double](m,m)
    dWfr = DenseMatrix.zeros[Double](m,m)
    dWor = DenseMatrix.zeros[Double](m,m)
    dWh_r = DenseMatrix.zeros[Double](m,m)
    dWix = DenseMatrix.zeros[Double](m,m)
    dWfx = DenseMatrix.zeros[Double](m,m)
    dWox = DenseMatrix.zeros[Double](m,m)
    dWh_x = DenseMatrix.zeros[Double](m,m)
    dr = DenseVector.zeros[Double](m)
    dC = DenseVector.zeros[Double](m)
    af.update()
  }
  def tanh(x:Double)={
    (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
 }
  def sigmoid(u:Double)={
    1/(1+math.exp(-u))
  }
 */

  val af = new Affine(m,n)
  val sig = new Sigmoid()
  val lstm = new LSTM(m,n)
  //学習
  def learning(){    
    val input = load("/home/share/text8")
    for(i<-0 until 1000){
      println("------"+i+"回目-------")
      var correct = 0
      var err = 0d
      var output = ""
      var answer = ""
      for(j<-0 until (input.size-15)/2){

        val tmp : Array[Double] = lstm.forward(input(j))
        val y1 : Array[Double] = sig.forward(af.forward(tmp))

        val y2 = sig.forward(af.forward(lstm.forward(input(j+1))))
        val y3 = sig.forward(af.forward(lstm.forward(input(j+2))))
        val y4 = sig.forward(af.forward(lstm.forward(input(j+3))))
        val y5 = sig.forward(af.forward(lstm.forward(input(j+4))))
        val y6 = sig.forward(af.forward(lstm.forward(input(j+5))))
        val y7 = sig.forward(af.forward(lstm.forward(input(j+6))))
        val y8 = sig.forward(af.forward(lstm.forward(input(j+7))))
        val y9 = sig.forward(af.forward(lstm.forward(input(j+8))))
        val y10 = sig.forward(af.forward(lstm.forward(input(j+9))))
        val y11 = sig.forward(af.forward(lstm.forward(input(j+10))))
        val y12 = sig.forward(af.forward(lstm.forward(input(j+11))))
        val y13 = sig.forward(af.forward(lstm.forward(input(j+12))))
        val y14 = sig.forward(af.forward(lstm.forward(input(j+13))))
        val y15 = sig.forward(af.forward(lstm.forward(input(j+14))))
              
        if(argmax(y15) == argmax(input(j+15))){
          correct += 1
        }
        err += - math.log(y15(argmax(input(j+15))))

        output += target(argmax(y15))
        answer += target(argmax(input(j+15)))

        lstm.backward(af.backward(sig.backward(y15-input(j+15))))
        lstm.backward(af.backward(sig.backward(y14-input(j+14))))
        lstm.backward(af.backward(sig.backward(y13-input(j+13))))
        lstm.backward(af.backward(sig.backward(y12-input(j+12))))
        lstm.backward(af.backward(sig.backward(y11-input(j+11))))
        lstm.backward(af.backward(sig.backward(y10-input(j+10))))
        lstm.backward(af.backward(sig.backward(y9-input(j+9))))
        lstm.backward(af.backward(sig.backward(y8-input(j+8))))
        lstm.backward(af.backward(sig.backward(y7-input(j+7))))
        lstm.backward(af.backward(sig.backward(y6-input(j+6))))
        lstm.backward(af.backward(sig.backward(y5-input(j+5))))
        lstm.backward(af.backward(sig.backward(y4-input(j+4))))
        lstm.backward(af.backward(sig.backward(y3-input(j+3))))
        lstm.backward(af.backward(sig.backward(y2-input(j+2))))
        lstm.backward(af.backward(sig.backward(y1-input(j+1))))
 
        lstm.update()
        af.update()
      }

      println("正解率"+((correct/((input.size-5)/2).toDouble)*100).toInt)
      println("誤差"+err/((input.size-15)/2))
      println("<output>")
      println(output)
      //println("<answer>")
      //println(answer)
      //println("")
    }

    //test
    var correct2 = 0
    var err2 = 0d
    var output2 = ""
    var answer2 = ""
    for(j <- (input.size-15)/2 until input.size-15){

      val y1 = sig.forward(af.forward(lstm.forward(input(j))))
      val y2 = sig.forward(af.forward(lstm.forward(input(j+1))))
      val y3 = sig.forward(af.forward(lstm.forward(input(j+2))))
      val y4 = sig.forward(af.forward(lstm.forward(input(j+3))))
      val y5 = sig.forward(af.forward(lstm.forward(input(j+4))))
      val y6 = sig.forward(af.forward(lstm.forward(input(j+5))))
      val y7 = sig.forward(af.forward(lstm.forward(input(j+6))))
      val y8 = sig.forward(af.forward(lstm.forward(input(j+7))))
      val y9 = sig.forward(af.forward(lstm.forward(input(j+8))))
      val y10 = sig.forward(af.forward(lstm.forward(input(j+9))))
      val y11 = sig.forward(af.forward(lstm.forward(input(j+10))))
      val y12 = sig.forward(af.forward(lstm.forward(input(j+11))))
      val y13 = sig.forward(af.forward(lstm.forward(input(j+12))))
      val y14 = sig.forward(af.forward(lstm.forward(input(j+13))))
      val y15 = sig.forward(af.forward(lstm.forward(input(j+14))))
      
      if(argmax(y15) == argmax(input(j+15))){
        correct2 += 1
      }
      err2 += - math.log(y15(argmax(input(j+15))))

      output2 += target(argmax(y15))
      answer2 += target(argmax(input(j+15)))

    }

    println("------test-------")
    println("correct:" + ((correct2/((input.size-15)/2).toDouble)*100).toInt + "%")
    println("err:" + err2/((input.size-15)/2))
    println("<output>")
    println(output2)
    println("<answer>")
    println(answer2)
    println("")

  }

  def load(fn:String)={
    val file = io.Source.fromFile(fn).getLines.toArray.head.split(" ").take(110)
    val ws = file.mkString(" ")
    println("ws:" + ws)
    ws.map(conv).toArray
  }

  def conv(c:Char)={
    val x = DenseVector.zeros[Double](m)
    x(target.indexOf(c)) = 1d
    x.toArray
  }

}

