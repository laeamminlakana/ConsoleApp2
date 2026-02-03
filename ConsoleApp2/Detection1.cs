using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Features2D;
using Emgu.CV.Linemod;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.IO;
using System.Text.Json;
using System.Drawing;

namespace ConsoleApp2;
public class Detection1
{
    static void Main(string[] args)
    {

        var cfgPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "detection", "yolov3.cfg");  // määrittelee cfg/weights tiedostojen sijainnit, ja tekee niistä luettavampia
        var weightsPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "detection", "yolov3.weights"); // sekä antaa virhe viestin mikäli tiedostoa ei löydä
        var namesPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "detection", "coco.names");

        if (!File.Exists(cfgPath) || !File.Exists(weightsPath) || !File.Exists(namesPath))
            throw new FileNotFoundException($"Missing files: {cfgPath}, {weightsPath}, {namesPath}");

        var net = DnnInvoke.ReadNet(cfgPath, weightsPath);
        var classLabels = File.ReadAllLines(namesPath);

        net.SetPreferableBackend(Emgu.CV.Dnn.Backend.OpenCV);
        net.SetPreferableTarget(Emgu.CV.Dnn.Target.Cpu);

        var vc = new VideoCapture(0, VideoCapture.API.DShow);

        Mat frame = new();
        VectorOfMat output = new();

        VectorOfPoint aimTarget = new();
        VectorOfRect boxes = new();
        VectorOfFloat scores = new();
        VectorOfInt indices = new();

        while (true)
        {
            vc.Read(frame);

            CvInvoke.Resize(frame, frame, new System.Drawing.Size(0, 0), .4, .4);

            boxes = new();
            indices = new();
            scores = new();
            aimTarget = new();

            var image = frame.ToImage<Bgr, byte>();

            var input = DnnInvoke.BlobFromImage(image, 1 / 255.0, swapRB: true);

            net.SetInput(input);

            net.Forward(output, net.UnconnectedOutLayersNames);

            for (int i = 0; i < output.Size; i++)
            {
                var mat = output[i];
                var data = (float[,])mat.GetData();

                for (int j = 0; j < data.GetLength(0); j++)
                {
                    float[] row = Enumerable.Range(0, data.GetLength(1))
                                  .Select(x => data[j, x])
                                  .ToArray();

                    var rowScore = row.Skip(5).ToArray();
                    var classId = rowScore.ToList().IndexOf(rowScore.Max());
                    var confidence = rowScore[classId];

                    if (confidence > 0.8f)
                    {
                        var centerX = (int)(row[0] * frame.Width);
                        var centerY = (int)(row[1] * frame.Height);
                        var boxWidth = (int)(row[2] * frame.Width);
                        var boxHeight = (int)(row[3] * frame.Height);

                        var x = (int)(centerX - (boxWidth / 2));
                        var y = (int)(centerY - (boxHeight / 2));

                        boxes.Push(new System.Drawing.Rectangle[] { new System.Drawing.Rectangle(x, y, boxWidth, boxHeight) });
                        indices.Push(new int[] { classId });
                        scores.Push(new float[] { confidence });
                        aimTarget.Push(new System.Drawing.Point[] { new System.Drawing.Point(centerX, centerY) });
                    }

                }

            }

            var bestIndex = DnnInvoke.NMSBoxes(boxes.ToArray(), scores.ToArray(), .8f, .8f);

            var frameOut = frame.ToImage<Bgr, byte>();

            for (int i = 0; i < bestIndex.Length; i++)
            {
                int index = bestIndex[i];
                var box = boxes[index];
                var bob = aimTarget[i];
                CvInvoke.Rectangle(frameOut, box, new MCvScalar(0, 255, 0), 1);
                CvInvoke.DrawMarker(frameOut, bob, new MCvScalar(255, 255, 255), Emgu.CV.CvEnum.MarkerTypes.Cross, 1, 1);

                CvInvoke.PutText(frameOut, classLabels[indices[index]], new System.Drawing.Point(box.X, box.Y - 20),
                Emgu.CV.CvEnum.FontFace.HersheyPlain, 1.0, new MCvScalar(0, 0, 255), 1);

            }

            CvInvoke.Resize(frameOut, frameOut, new System.Drawing.Size(0, 0), 4, 4);
            CvInvoke.Imshow("output", frameOut);

            // Convert aimTarget to array of points and process only if there is at least one point.
            System.Drawing.Point[] points = aimTarget.ToArray();
            if (points.Length > 0)
            {
                var aimpoint = points[0];

                if (aimpoint.X != 0 && aimpoint.Y != 0)
                {
                    var jsonstring1 = System.Text.Json.JsonSerializer.Serialize(aimpoint);
                    File.WriteAllText("aimTarget.json", jsonstring1);
                    Task.Run(() => Laser.Main_Laser(args));
                }


            }
            if (CvInvoke.WaitKey(1) == 27)
                break;
        }
    }
}

