using System;
using System.IO;
using System.Text.Json;
using System.Drawing;
namespace ConsoleApp2;



public static class Laser
{
	
    public static void Main_Laser(string[] args)
	{
        const string path = "aimTarget.json";
        string jsonstring;
            jsonstring = File.ReadAllText(path);

            Point kohde = JsonSerializer.Deserialize<Point>(jsonstring);
            Console.WriteLine("Laser targeting point: X" + kohde.X + " and " + kohde.Y);

    }
}
