using Merlin.CallEval.AI;
using Merlin.CallEval.Settings;
using Merlin.CallEval.Utilities;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text;
using System.Threading.Tasks;
using Volo.Abp.DependencyInjection;
using Volo.Abp.SettingManagement;

namespace Merlin.CallEval.Jobs.Transcription
{
    /// <summary>
    /// Transcription provider for Together AI API
    /// </summary>
    public class TogetherTranscriptionProvider : ITranscriptionProvider, ITransientDependency
    {
        private const string TogetherTranscriptionUrl = "https://api.together.ai/v1/audio/transcriptions";
        private const string DefaultModel = "openai/whisper-large-v3";        
        private const string ResponseFormat = "verbose_json";
        private const string Boundary = "---------------------------24838421832148";
        private const string NewLine = "\r\n";

        private readonly ISettingManager _settingManager;

        public TogetherTranscriptionProvider(ISettingManager settingManager)
        {
            _settingManager = settingManager;
        }

        public async Task<TranscriptionResult> TranscribeAsync(Stream audioStream, string blobName)
        {
            try
            {
                var apiKey = await _settingManager.GetOrNullForCurrentTenantAsync(CallEvalSettingNames.Transcription.OpenAiApiKey);
                var expectedLanguage = await _settingManager.GetOrNullForCurrentTenantAsync(CallEvalSettingNames.Transcription.ExpectedLanguage);

                if (string.IsNullOrEmpty(apiKey))
                    return TranscriptionResult.Failure("Together AI API key is not configured");

                using var memoryStream = BuildMultipartContent(audioStream, blobName, expectedLanguage);

                using var client = new HttpClient();                
                client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
                
                using var requestContent = new StreamContent(memoryStream);
                requestContent.Headers.ContentType = new MediaTypeHeaderValue("multipart/form-data");
                requestContent.Headers.ContentType.Parameters.Add(new NameValueHeaderValue("boundary", Boundary));

                var response = await client.PostAsync(TogetherTranscriptionUrl, requestContent);
                var str = await response.Content.ReadAsStringAsync();
                response.EnsureSuccessStatusCode();

                var result = await response.Content.ReadFromJsonAsync<TogetherWhisperResponseDto>();

                if (result == null)
                    return TranscriptionResult.Failure("Failed to parse Together AI response");

                return ConvertToTranscriptionResult(result);
            }
            catch (Exception ex)
            {
                return TranscriptionResult.Failure($"Together AI transcription failed: {ex.Message}");
            }
        }

        private MemoryStream BuildMultipartContent(Stream audioStream, string blobName,string language)
        {
            var memoryStream = new MemoryStream();
            var writer = new StreamWriter(memoryStream, Encoding.ASCII);

            WriteField(writer, "model", DefaultModel);
            WriteField(writer, "language", language);
            WriteField(writer, "response_format", ResponseFormat);
            WriteField(writer, "diarize", "true");

            // Write file header
            writer.Write($"--{Boundary}{NewLine}");
            writer.Write($"Content-Disposition: form-data; name=\"file\"; filename=\"{blobName}\"{NewLine}");
            writer.Write($"Content-Type: audio/wav{NewLine}{NewLine}");
            writer.Flush();

            // Write file content
            var fileBytes = audioStream.GetAllBytes();
            memoryStream.Write(fileBytes, 0, fileBytes.Length);

            writer.Write(NewLine);
            writer.Write($"--{Boundary}--{NewLine}");
            writer.Flush();

            memoryStream.Position = 0;
            return memoryStream;
        }

        private void WriteField(StreamWriter writer, string name, string value)
        {
            writer.Write($"--{Boundary}{NewLine}");
            writer.Write($"Content-Disposition: form-data; name=\"{name}\"{NewLine}{NewLine}");
            writer.Write(value);
            writer.Write(NewLine);
        }

        private TranscriptionResult ConvertToTranscriptionResult(TogetherWhisperResponseDto response)
        {
            var segments = response.SpeakerSegments?.Select(s => new TranscriptionSegment
            {
                Text = s.Text ?? string.Empty,
                Speaker = s.SpeakerId ?? string.Empty,
                Start = s.Start,
                End = s.End
            }).ToList() ?? new List<TranscriptionSegment>();

            var silenceDuration = CalculateSilenceDuration(response.Segments);

            return TranscriptionResult.Success(
                segments,
                response.Duration,
                silenceDuration
            );
        }

        private int CalculateSilenceDuration(List<TogetherSegmentDto> segments)
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
                {
                    totalSilenceDuration += silenceDuration;
                }
            }
            return totalSilenceDuration;
        }
    }
}
