using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Text;
using System.Web.Script.Serialization;
using System.Collections.Generic;
using System.Windows.Forms;

internal sealed class Launcher : Form {
    readonly string root=AppDomain.CurrentDomain.BaseDirectory;
    readonly Label status=new Label(), detail=new Label(), board=new Label();
    readonly ProgressBar progress=new ProgressBar();
    readonly Button retry=new Button(), open=new Button(), logs=new Button();
    readonly JavaScriptSerializer json=new JavaScriptSerializer();
    Process worker;
    StreamWriter logfile;
    string workbenchUrl="";
    bool running=false, ready=false, diagnosed=false;
    readonly bool diagnostic;
    public Launcher(bool diagnose) {
        diagnostic=diagnose;
        Text="CT Studio 上位机";
        Font=new Font("Microsoft YaHei UI",10F);
        AutoScaleMode=AutoScaleMode.None;
        ClientSize=new Size(660,430);
        FormBorderStyle=FormBorderStyle.FixedSingle;
        MaximizeBox=false;
        StartPosition=FormStartPosition.CenterScreen;
        BackColor=Color.FromArgb(245,248,252);
        try { Icon=new Icon(Path.Combine(root,"launcher","app.ico")); } catch {}
        var heading=new Label {Text="CT Studio",Font=new Font("Segoe UI",25F,FontStyle.Bold),ForeColor=Color.FromArgb(24,55,89),AutoSize=false,Location=new Point(30,24),Size=new Size(590,52),UseCompatibleTextRendering=true};
        Controls.Add(heading);
        var subtitle=new Label {Text="连接开发板，即可开始影像处理",AutoSize=false,Size=new Size(580,32),Location=new Point(33,78),ForeColor=Color.FromArgb(88,105,123),UseCompatibleTextRendering=true};
        Controls.Add(subtitle);
        var card=new Panel {Location=new Point(30,120),Size=new Size(600,190),BackColor=Color.White};
        Controls.Add(card);
        status.SetBounds(22,20,552,32);status.Text="准备启动…";status.Font=new Font(Font,FontStyle.Bold);
        card.Controls.Add(status);
        progress.SetBounds(24,65,550,7);progress.Style=ProgressBarStyle.Marquee;progress.MarqueeAnimationSpeed=25;card.Controls.Add(progress);
        detail.SetBounds(22,92,552,55);detail.Text="首次打开会检查内置环境，请稍候。";detail.ForeColor=Color.FromArgb(75,93,112);card.Controls.Add(detail);
        board.SetBounds(22,155,552,25);board.Text="开发板 192.168.2.99  ·  电脑端集成版";board.ForeColor=Color.FromArgb(108,123,138);board.Font=new Font(Font.FontFamily,9F);card.Controls.Add(board);
        retry.SetBounds(30,332,145,40);retry.Text="重新连接";retry.Enabled=false;StyleButton(retry,false);retry.Click+=(s,e)=>StartWorker();Controls.Add(retry);
        open.SetBounds(190,332,165,40);open.Text="打开处理界面";open.Enabled=false;StyleButton(open,true);open.Click+=(s,e)=>OpenWorkbench();Controls.Add(open);
        logs.SetBounds(478,332,152,40);logs.Text="查看日志";StyleButton(logs,false);logs.Click+=(s,e)=>OpenLogs();Controls.Add(logs);
        var note=new Label {Text="关闭此启动窗口后，网页中的处理任务仍会继续。",AutoSize=true,Location=new Point(32,390),Font=new Font(Font.FontFamily,9F),ForeColor=Color.FromArgb(108,123,138)};
        note.AutoSize=false;note.Size=new Size(590,30);note.UseCompatibleTextRendering=true;
        status.UseCompatibleTextRendering=true;detail.UseCompatibleTextRendering=true;board.UseCompatibleTextRendering=true;
        retry.UseCompatibleTextRendering=true;open.UseCompatibleTextRendering=true;logs.UseCompatibleTextRendering=true;
        Controls.Add(note);
        Shown+=(s,e)=>StartWorker();
        FormClosing+=(s,e)=>{
            if(running) {
                e.Cancel=true;
                detail.Text="正在检查连接，请稍候。连接结束后可关闭此窗口。";
            }
        };
    }
    void StyleButton(Button b,bool primary) {
        b.FlatStyle=FlatStyle.Flat;b.FlatAppearance.BorderColor=primary?Color.FromArgb(33,101,205):Color.FromArgb(207,217,229);
        b.BackColor=primary?Color.FromArgb(33,101,205):Color.White;b.ForeColor=primary?Color.White:Color.FromArgb(37,66,99);
    }
    void Ui(Action action) {if(!IsDisposed&&IsHandleCreated)try{BeginInvoke(action);}catch(InvalidOperationException){}}
    void StartWorker() {
        if(running)return;
        running=true;ready=false;diagnosed=false;open.Enabled=false;retry.Enabled=false;
        status.Text="正在启动上位机…";detail.Text="自动检查内置环境、开发板连接和处理界面。";
        progress.Style=ProgressBarStyle.Marquee;
        try {
            string python=Path.Combine(root,"runtime","python.exe"),script=Path.Combine(root,"app","desktop_bootstrap.py");
            if(!File.Exists(python)||!File.Exists(script))throw new FileNotFoundException("请先完整解压软件包，再双击上位机；不要只复制 exe 文件。");
            string logdir=Path.Combine(root,"app","data","desktop_launcher");Directory.CreateDirectory(logdir);
            logfile=new StreamWriter(Path.Combine(logdir,"desktop_"+DateTime.Now.ToString("yyyyMMdd_HHmmss_fff")+".log"),false,new UTF8Encoding(false));logfile.AutoFlush=true;
            var info=new ProcessStartInfo(python,"-B -u -X utf8 \""+script+"\""+(diagnostic?" --diagnose":"")) {
                WorkingDirectory=Path.Combine(root,"app"),UseShellExecute=false,CreateNoWindow=true,
                RedirectStandardOutput=true,RedirectStandardError=true,StandardOutputEncoding=Encoding.UTF8,StandardErrorEncoding=Encoding.UTF8
            };
            info.EnvironmentVariables.Remove("PYTHONHOME");info.EnvironmentVariables.Remove("PYTHONPATH");
            info.EnvironmentVariables["PYTHONNOUSERSITE"]="1";
            info.EnvironmentVariables["PATH"]=Path.Combine(root,"runtime")+";"+Path.Combine(root,"runtime","Library","bin")+";"+Environment.GetEnvironmentVariable("PATH");
            info.EnvironmentVariables["MPLCONFIGDIR"]=Path.Combine(logdir,"matplotlib");
            worker=new Process {StartInfo=info,EnableRaisingEvents=true};
            worker.OutputDataReceived+=(s,e)=>{if(e.Data!=null)Ui(()=>HandleLine(e.Data));};
            worker.ErrorDataReceived+=(s,e)=>{if(e.Data!=null)Ui(()=>WriteLog(e.Data));};
            worker.Exited+=(s,e)=>Ui(()=>WorkerExited());
            worker.Start();worker.BeginOutputReadLine();worker.BeginErrorReadLine();
        } catch(Exception ex) {Fail(ex.Message);}
    }
    void WriteLog(string value) {try{if(logfile!=null)logfile.WriteLine(value);}catch(IOException){}}
    void HandleLine(string line) {
        WriteLog(line);
        try {
            var data=json.Deserialize<Dictionary<string,object>>(line);
            string kind=Convert.ToString(data["event"]),message=Convert.ToString(data["message"]);
            if(kind=="progress") {status.Text=message;detail.Text="自动连接中，无需输入命令。请保持板子通电和网线连接。";}
            else if(kind=="ready") {
                var uri=new Uri(Convert.ToString(data["url"]));
                if(uri.Scheme!="http"||uri.Host!="127.0.0.1"||uri.Port<8099||uri.Port>8109)throw new InvalidDataException("处理界面地址异常。");
                workbenchUrl=uri.AbsoluteUri;ready=true;status.Text="已就绪";detail.Text=message+"\n处理界面已自动打开，也可以点击下方按钮再次打开。";
                progress.Style=ProgressBarStyle.Continuous;progress.Value=100;open.Enabled=true;board.Text="开发板已连接  ·  "+workbenchUrl;
                OpenWorkbench();
            } else if(kind=="diagnosed") {
                diagnosed=true;status.Text="本机环境检查通过";detail.Text=message;
                progress.Style=ProgressBarStyle.Continuous;progress.Value=100;
            } else if(kind=="error") {Fail(message);}
        } catch(Exception ex) {if(line.StartsWith("{"))WriteLog("Launcher event: "+ex.GetType().Name);}
    }
    void WorkerExited() {
        // Defer one UI queue cycle so the final redirected JSON line is consumed first.
        var timer=new Timer();timer.Interval=300;timer.Tick+=(s,e)=>{
            timer.Stop();timer.Dispose();running=false;retry.Enabled=true;
            if(!ready&&!diagnosed&&status.Text!="连接尚未完成")Fail("启动未完成。请点击“重新连接”；仍有问题可打开日志查看。");
            if(logfile!=null){logfile.Dispose();logfile=null;}
        };timer.Start();
    }
    void Fail(string message) {
        running=false;status.Text="连接尚未完成";detail.Text=message;retry.Enabled=true;
        progress.Style=ProgressBarStyle.Continuous;progress.Value=0;WriteLog(message);
    }
    void OpenWorkbench() {
        if(String.IsNullOrEmpty(workbenchUrl))return;
        try {Process.Start(new ProcessStartInfo(workbenchUrl){UseShellExecute=true});}
        catch(Exception){detail.Text="处理界面已就绪，请在浏览器打开 "+workbenchUrl;}
    }
    void OpenLogs() {
        string dir=Path.Combine(root,"app","data","desktop_launcher");
        try{Directory.CreateDirectory(dir);Process.Start(new ProcessStartInfo("explorer.exe","\""+dir+"\""){UseShellExecute=true});}
        catch(Exception ex){MessageBox.Show(ex.Message,"日志目录");}
    }
    [STAThread] public static void Main(string[] args) {
        Application.EnableVisualStyles();Application.SetCompatibleTextRenderingDefault(false);
        using(var mutex=new System.Threading.Mutex(false,"Local\\CTStudioDesktop_"+StableId(AppDomain.CurrentDomain.BaseDirectory))) {
            bool held=false;
            try{held=mutex.WaitOne(0);}catch(System.Threading.AbandonedMutexException){held=true;}
            if(!held){MessageBox.Show("上位机启动窗口已经打开，请切换到原窗口。","CT Studio");return;}
            try{Application.Run(new Launcher(Array.IndexOf(args,"--diagnose")>=0));}
            finally{mutex.ReleaseMutex();}
        }
    }
    static string StableId(string text) {
        using(var hash=System.Security.Cryptography.SHA256.Create())return BitConverter.ToString(hash.ComputeHash(Encoding.UTF8.GetBytes(text.ToLowerInvariant()))).Replace("-","").Substring(0,20);
    }
}
