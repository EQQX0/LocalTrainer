// One-click installer: LocalTrainer-Setup-<version>.exe
//
// Built by tools/build_installer.py with the C# 5 compiler that ships with .NET Framework 4 (csc.exe), so it
// runs on any Windows 10/11 without extra runtimes. The release zip is embedded as the "payload.zip" resource.
//
// Run with no arguments it installs to %LOCALAPPDATA%\Programs\LocalTrainer; the first argument picks another
// folder. A folder that already holds LocalTrainer keeps every file (projects, settings, .venv): the installer
// only finishes setup there if needed and starts the app, and the in-app updater handles newer versions.
using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Reflection;

static class Setup
{
    const string Payload = "payload.zip";

    static int Main(string[] args)
    {
        Console.Title = "LocalTrainer Setup";
        string target = args.Length > 0 ? Path.GetFullPath(args[0]) : Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "LocalTrainer");
        try
        {
            Console.WriteLine();
            Console.WriteLine(" LocalTrainer Setup " + Assembly.GetExecutingAssembly().GetName().Version.ToString(3));
            Console.WriteLine(" ------------------------");
            if (File.Exists(Path.Combine(target, "main.py")))
            {
                Console.WriteLine(" LocalTrainer is already installed in " + target);
                Console.WriteLine(" Your projects and settings stay as they are.");
            }
            else
            {
                Console.WriteLine(" Installing to " + target);
                Extract(target);
            }
            Console.WriteLine();

            if (!PackagesReady(target))
            {
                int code = Run("cmd.exe", "/d /c \"\"" + Path.Combine(target, "setup.bat") + "\" /nopause\"", target);
                if (code != 0 || !PackagesReady(target))
                    return Fail("Setup did not finish. Read the messages above, then run this installer again.");
            }

            string scripts = Path.Combine(target, ".venv", "Scripts");
            string pythonw = Path.Combine(scripts, "pythonw.exe");
            var start = new ProcessStartInfo(File.Exists(pythonw) ? pythonw : Path.Combine(scripts, "python.exe"),
                                             Quote(Path.Combine(target, "main.py")));
            start.WorkingDirectory = target;
            start.UseShellExecute = false;
            Process.Start(start);
            Console.WriteLine(" Starting LocalTrainer. Next time, open it from the desktop shortcut.");
            System.Threading.Thread.Sleep(2500);
            return 0;
        }
        catch (Exception e)
        {
            return Fail(e.Message);
        }
    }

    // Unpacks into "<target>.partial" first, so a failed or cancelled run never leaves a half-written install.
    static void Extract(string target)
    {
        if (Directory.Exists(target) && Directory.EnumerateFileSystemEntries(target).Any())
            throw new IOException(target + " is not empty and holds no LocalTrainer install. " +
                                  "Empty it, or run the installer with another folder: LocalTrainer-Setup.exe \"D:\\Apps\\LocalTrainer\"");
        string partial = target + ".partial";
        if (Directory.Exists(partial))
            Directory.Delete(partial, true);
        Directory.CreateDirectory(partial);
        string root = Path.GetFullPath(partial) + Path.DirectorySeparatorChar;

        using (Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(Payload))
        using (var zip = new ZipArchive(stream, ZipArchiveMode.Read))
        {
            int count = 0;
            foreach (ZipArchiveEntry entry in zip.Entries)
            {
                // entries sit under "LocalTrainer-<version>/"; drop that folder
                int slash = entry.FullName.IndexOf('/');
                string rel = slash < 0 ? "" : entry.FullName.Substring(slash + 1);
                if (rel.Length == 0 || rel.EndsWith("/"))
                    continue;
                string dest = Path.GetFullPath(Path.Combine(partial, rel.Replace('/', Path.DirectorySeparatorChar)));
                if (!dest.StartsWith(root, StringComparison.OrdinalIgnoreCase))
                    throw new IOException("The installer holds an unsafe path: " + entry.FullName);
                Directory.CreateDirectory(Path.GetDirectoryName(dest));
                entry.ExtractToFile(dest, true);
                count++;
            }
            Console.WriteLine(" Unpacked " + count + " files.");
        }

        if (Directory.Exists(target))
            Directory.Delete(target);  // empty, checked above
        Directory.CreateDirectory(Path.GetDirectoryName(target));
        Directory.Move(partial, target);
    }

    // True when .venv exists and imports the heavy packages, so a setup that stopped halfway runs again.
    static bool PackagesReady(string target)
    {
        string python = Path.Combine(target, ".venv", "Scripts", "python.exe");
        if (!File.Exists(python))
            return false;
        var info = new ProcessStartInfo(python, "-c \"import torch, PyQt6, ultralytics\"");
        info.WorkingDirectory = target;
        info.UseShellExecute = false;
        info.CreateNoWindow = true;  // no console: an ImportError traceback goes nowhere
        using (Process p = Process.Start(info))
        {
            p.WaitForExit();
            return p.ExitCode == 0;
        }
    }

    // Runs in this console window, so pip's progress shows here.
    static int Run(string file, string arguments, string dir)
    {
        var info = new ProcessStartInfo(file, arguments);
        info.WorkingDirectory = dir;
        info.UseShellExecute = false;
        using (Process p = Process.Start(info))
        {
            p.WaitForExit();
            return p.ExitCode;
        }
    }

    static string Quote(string path)
    {
        return "\"" + path + "\"";
    }

    static int Fail(string message)
    {
        Console.WriteLine();
        Console.WriteLine(" " + message);
        if (!Console.IsInputRedirected)
        {
            Console.WriteLine(" Press any key to close this window.");
            Console.ReadKey(true);
        }
        return 1;
    }
}
