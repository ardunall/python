using System.Diagnostics;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

// ==================== CONFIGURATION ====================
var TogetherApiKey = LoadEnvValue("TogetherApiKey");
const string ExpectedLanguage = "tr";  // Dil kodu (tr, en, etc.)
const string AudioFilePath = "audio/audio5.wav";  // Ses dosyası yolu
const string PythonScriptPath = "python/main.py";  // Python script yolu
// ========================================================

const string TogetherTranscriptionUrl = "https://api.together.ai/v1/audio/transcriptions";
const string DefaultModel = "openai/whisper-large-v3";
const string ResponseFormat = "verbose_json";
const string Boundary = "---------------------------24838421832148";
const string NewLine = "\r\n";

await RunAsync();

async Task RunAsync()
{
    Console.WriteLine("=== Together AI Transcription + Speaker Identification ===\n");

    // 1. Ses dosyasını transcribe et
    Console.WriteLine($"[1] Ses dosyası transcribe ediliyor: {AudioFilePath}");
    var transcriptionResult = await TranscribeAsync(AudioFilePath);

    if (!transcriptionResult.IsSuccess)
    {
        Console.WriteLine($"HATA: {transcriptionResult.ErrorMessage}");
        return;
    }

    Console.WriteLine($"[✓] Transcription tamamlandı! Süre: {transcriptionResult.Duration:F2} saniye");
    Console.WriteLine($"[✓] {transcriptionResult.Segments.Count} segment bulundu\n");

    // 2. Segmentleri Python formatına dönüştür
    var pythonInput = transcriptionResult.Segments.Select(s => new
    {
        speaker = s.Speaker,
        text = s.Text
    }).ToList();

    var jsonInput = JsonSerializer.Serialize(pythonInput, new JsonSerializerOptions { WriteIndented = true });
    Console.WriteLine("[2] Python'a gönderilecek veri:");
    Console.WriteLine(jsonInput);

    // 3. Python script'i çağır
    Console.WriteLine($"\n[3] Python script çağrılıyor: {PythonScriptPath}");
    var speakerRoles = await CallPythonAsync(jsonInput);

    Console.WriteLine("\n[✓] Sonuç:");
    Console.WriteLine(speakerRoles);
}

async Task<TranscriptionResult> TranscribeAsync(string audioFilePath)
{
    try
    {
        if (!File.Exists(audioFilePath))
            return TranscriptionResult.Failure($"Ses dosyası bulunamadı: {audioFilePath}");

        if (string.IsNullOrEmpty(TogetherApiKey) || TogetherApiKey == "YOUR_TOGETHER_API_KEY_HERE")
            return TranscriptionResult.Failure("Together AI API anahtarı yapılandırılmamış!");

        using var audioStream = File.OpenRead(audioFilePath);
        using var memoryStream = BuildMultipartContent(audioStream, Path.GetFileName(audioFilePath), ExpectedLanguage);

        using var client = new HttpClient();
        client.Timeout = TimeSpan.FromMinutes(5);
        client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", TogetherApiKey);

        using var requestContent = new StreamContent(memoryStream);
        requestContent.Headers.ContentType = new MediaTypeHeaderValue("multipart/form-data");
        requestContent.Headers.ContentType.Parameters.Add(new NameValueHeaderValue("boundary", Boundary));

        var response = await client.PostAsync(TogetherTranscriptionUrl, requestContent);
        var responseStr = await response.Content.ReadAsStringAsync();

        if (!response.IsSuccessStatusCode)
            return TranscriptionResult.Failure($"API Hatası: {response.StatusCode} - {responseStr}");

        var result = JsonSerializer.Deserialize<TogetherWhisperResponseDto>(responseStr);

        if (result == null)
            return TranscriptionResult.Failure("Together AI yanıtı parse edilemedi");

        return ConvertToTranscriptionResult(result);
    }
    catch (Exception ex)
    {
        return TranscriptionResult.Failure($"Transcription hatası: {ex.Message}");
    }
}

MemoryStream BuildMultipartContent(Stream audioStream, string fileName, string language)
{
    var memoryStream = new MemoryStream();
    var writer = new StreamWriter(memoryStream, Encoding.ASCII);

    WriteField(writer, "model", DefaultModel);
    WriteField(writer, "language", language);
    WriteField(writer, "response_format", ResponseFormat);
    WriteField(writer, "diarize", "true");

    // Write file header
    writer.Write($"--{Boundary}{NewLine}");
    writer.Write($"Content-Disposition: form-data; name=\"file\"; filename=\"{fileName}\"{NewLine}");
    writer.Write($"Content-Type: audio/wav{NewLine}{NewLine}");
    writer.Flush();

    // Write file content
    using var ms = new MemoryStream();
    audioStream.CopyTo(ms);
    var fileBytes = ms.ToArray();
    memoryStream.Write(fileBytes, 0, fileBytes.Length);

    writer.Write(NewLine);
    writer.Write($"--{Boundary}--{NewLine}");
    writer.Flush();

    memoryStream.Position = 0;
    return memoryStream;
}

void WriteField(StreamWriter writer, string name, string value)
{
    writer.Write($"--{Boundary}{NewLine}");
    writer.Write($"Content-Disposition: form-data; name=\"{name}\"{NewLine}{NewLine}");
    writer.Write(value);
    writer.Write(NewLine);
}

TranscriptionResult ConvertToTranscriptionResult(TogetherWhisperResponseDto response)
{
    var segments = response.SpeakerSegments?.Select(s => new TranscriptionSegment
    {
        Text = s.Text ?? string.Empty,
        Speaker = s.SpeakerId ?? "Speaker_0",
        Start = s.Start,
        End = s.End
    }).ToList() ?? new List<TranscriptionSegment>();

    var silenceDuration = CalculateSilenceDuration(response.Segments);

    return TranscriptionResult.Success(segments, response.Duration, silenceDuration);
}

int CalculateSilenceDuration(List<TogetherSegmentDto>? segments)
{
    if (segments == null || segments.Count < 2)
        return 0;

    int totalSilenceDuration = 0;
    for (int i = 1; i < segments.Count; i++)
    {
        var previousSegmentEnd = segments[i - 1].End;
        var currentSegmentStart = segments[i].Start;

        var silenceDuration = (int)(currentSegmentStart - previousSegmentEnd);
        if (silenceDuration > 0)
            totalSilenceDuration += silenceDuration;
    }
    return totalSilenceDuration;
}

async Task<string> CallPythonAsync(string jsonInput)
{
    try
    {
        // JSON'u geçici dosyaya yaz
        var tempJsonPath = Path.GetTempFileName();
        await File.WriteAllTextAsync(tempJsonPath, jsonInput);

        var psi = new ProcessStartInfo
        {
            FileName = "uv",
            Arguments = $"run --project python \"{PythonScriptPath}\" \"{tempJsonPath}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = AppContext.BaseDirectory.Contains("bin") 
                ? Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."))
                : Environment.CurrentDirectory
        };

        using var process = Process.Start(psi);
        if (process == null)
            return "Python process başlatılamadı!";

        // Read stdout and stderr concurrently to avoid deadlock
        var outputTask = process.StandardOutput.ReadToEndAsync();
        var errorTask = process.StandardError.ReadToEndAsync();
        
        await process.WaitForExitAsync();
        
        var output = await outputTask;
        var error = await errorTask;

        // Geçici dosyayı sil
        File.Delete(tempJsonPath);

        if (!string.IsNullOrEmpty(error) && process.ExitCode != 0)
            return $"Python Hatası: {error}";

        return output;
    }
    catch (Exception ex)
    {
        return $"Python çağrısı hatası: {ex.Message}";
    }
}

// ==================== Helper Functions ====================

string LoadEnvValue(string key)
{
    var envPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".env");
    if (!File.Exists(envPath))
        envPath = ".env"; // Fallback to current directory
    
    if (File.Exists(envPath))
    {
        foreach (var line in File.ReadAllLines(envPath))
        {
            var parts = line.Split('=', 2);
            if (parts.Length == 2 && parts[0].Trim() == key)
                return parts[1].Trim();
        }
    }
    
    // Fallback to environment variable
    return Environment.GetEnvironmentVariable(key) ?? string.Empty;
}

// ==================== DTO Classes ====================

public class TranscriptionResult
{
    public bool IsSuccess { get; set; }
    public string? ErrorMessage { get; set; }
    public List<TranscriptionSegment> Segments { get; set; } = new();
    public double Duration { get; set; }
    public int SilenceDuration { get; set; }

    public static TranscriptionResult Success(List<TranscriptionSegment> segments, double duration, int silenceDuration)
        => new() { IsSuccess = true, Segments = segments, Duration = duration, SilenceDuration = silenceDuration };

    public static TranscriptionResult Failure(string message)
        => new() { IsSuccess = false, ErrorMessage = message };
}

public class TranscriptionSegment
{
    public string Text { get; set; } = string.Empty;
    public string Speaker { get; set; } = string.Empty;
    public double Start { get; set; }
    public double End { get; set; }
}

public class TogetherWhisperResponseDto
{
    [JsonPropertyName("text")]
    public string? Text { get; set; }

    [JsonPropertyName("duration")]
    public double Duration { get; set; }

    [JsonPropertyName("segments")]
    public List<TogetherSegmentDto>? Segments { get; set; }

    [JsonPropertyName("speaker_segments")]
    public List<TogetherSpeakerSegmentDto>? SpeakerSegments { get; set; }
}

public class TogetherSegmentDto
{
    [JsonPropertyName("start")]
    public double Start { get; set; }

    [JsonPropertyName("end")]
    public double End { get; set; }

    [JsonPropertyName("text")]
    public string? Text { get; set; }
}

public class TogetherSpeakerSegmentDto
{
    [JsonPropertyName("start")]
    public double Start { get; set; }

    [JsonPropertyName("end")]
    public double End { get; set; }

    [JsonPropertyName("text")]
    public string? Text { get; set; }

    [JsonPropertyName("speaker_id")]
    public string? SpeakerId { get; set; }
}
