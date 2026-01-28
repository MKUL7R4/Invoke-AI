function Invoke-AI {
    <#
    .SYNOPSIS
        Calls various AI APIs with unified interface
    
    .DESCRIPTION
        A comprehensive PowerShell function that can call multiple AI provider APIs including OpenAI, Anthropic Claude, Google Gemini, and others with a unified interface.
    
    .PARAMETER Provider
        The AI provider to use. Supported values: OpenAI, Anthropic, Google, Azure, Cohere, HuggingFace
    
    .PARAMETER ApiKey
        API key for the selected provider. Can also be set via environment variables.
    
    .PARAMETER Prompt
        The text prompt to send to the AI
    
    .PARAMETER Model
        The specific model to use. Defaults to provider's standard model if not specified.
    
    .PARAMETER MaxTokens
        Maximum number of tokens in the response (default: 1000)
    
    .PARAMETER Temperature
        Controls randomness (0.0-2.0, default: 0.7)
    
    .PARAMETER SystemPrompt
        Optional system prompt for context setting
    
    .PARAMETER Endpoint
        Custom API endpoint (for Azure or custom deployments)
    
    .PARAMETER ConfigFile
        Path to JSON configuration file containing API keys and settings
    
    .EXAMPLE
        Invoke-AI -Provider OpenAI -Prompt "What is PowerShell?" -ApiKey "sk-..."
    
    .EXAMPLE
        Invoke-AI -Provider Anthropic -Prompt "Explain quantum computing" -ConfigFile "~\ai-config.json"
    
    .EXAMPLE
        Invoke-AI -Provider Google -Prompt "Write a Python function" -Model "gemini-pro" -Temperature 0.3
    #>
    
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("OpenAI", "Anthropic", "Google", "Azure", "Cohere", "HuggingFace")]
        [string]$Provider,
        
        [Parameter(Mandatory = $false)]
        [string]$ApiKey,
        
        [Parameter(Mandatory = $true)]
        [string]$Prompt,
        
        [Parameter(Mandatory = $false)]
        [string]$Model,
        
        [Parameter(Mandatory = $false)]
        [int]$MaxTokens = 1000,
        
        [Parameter(Mandatory = $false)]
        [ValidateRange(0.0, 2.0)]
        [double]$Temperature = 0.7,
        
        [Parameter(Mandatory = $false)]
        [string]$SystemPrompt,
        
        [Parameter(Mandatory = $false)]
        [string]$Endpoint,
        
        [Parameter(Mandatory = $false)]
        [string]$ConfigFile,
        
        [Parameter(Mandatory = $false)]
        [switch]$FullResponse,
        
        [Parameter(Mandatory = $false)]
        [switch]$ResponseOnly
    )
    
    # Error handling
    $ErrorActionPreference = "Stop"
    
    try {
        # Load configuration file if provided
        if ($ConfigFile -and (Test-Path $ConfigFile)) {
            $config = Get-Content $ConfigFile | ConvertFrom-Json
            if (-not $ApiKey -and $config.$Provider.ApiKey) {
                $ApiKey = $config.$Provider.ApiKey
            }
            if (-not $Endpoint -and $config.$Provider.Endpoint) {
                $Endpoint = $config.$Provider.Endpoint
            }
            if (-not $Model -and $config.$Provider.Model) {
                $Model = $config.$Provider.Model
            }
        }
        
        # Get API key from environment variable if not provided
        if (-not $ApiKey) {
            $envVar = switch ($Provider) {
                "OpenAI" { "OPENAI_API_KEY" }
                "Anthropic" { "ANTHROPIC_API_KEY" }
                "Google" { "GOOGLE_AI_API_KEY" }
                "Azure" { "AZURE_OPENAI_API_KEY" }
                "Cohere" { "COHERE_API_KEY" }
                "HuggingFace" { "HUGGINGFACE_API_KEY" }
            }
            $ApiKey = [Environment]::GetEnvironmentVariable($envVar)
        }
        
        if (-not $ApiKey) {
            throw "API key not provided. Use -ApiKey parameter, set environment variable, or include in config file."
        }
        
        # Set default models if not specified
        if (-not $Model) {
            $Model = switch ($Provider) {
                "OpenAI" { "gpt-3.5-turbo" }
                "Anthropic" { "claude-3-sonnet-20240229" }
                "Google" { "gemini-2.5-flash" }
                "Azure" { "gpt-35-turbo" }
                "Cohere" { "command" }
                "HuggingFace" { "microsoft/DialoGPT-medium" }
            }
        }
        
        # Set default endpoints if not specified
        if (-not $Endpoint) {
            $Endpoint = switch ($Provider) {
                "OpenAI" { "https://api.openai.com/v1/chat/completions" }
                "Anthropic" { "https://api.anthropic.com/v1/messages" }
                "Google" { "https://generativelanguage.googleapis.com/v1/models/$($Model):generateContent" }
                "Azure" { throw "Azure requires custom endpoint. Use -Endpoint parameter." }
                "Cohere" { "https://api.cohere.ai/v1/generate" }
                "HuggingFace" { "https://api-inference.huggingface.co/models/$Model" }
            }
        }
        
        # Build request body based on provider
        $body = switch ($Provider) {
            "OpenAI" {
                $messages = @()
                if ($SystemPrompt) {
                    $messages += @{ role = "system"; content = $SystemPrompt }
                }
                $messages += @{ role = "user"; content = $Prompt }
                
                @{
                    model       = $Model
                    messages    = $messages
                    max_tokens  = $MaxTokens
                    temperature = $Temperature
                } | ConvertTo-Json -Depth 10
            }
            
            "Anthropic" {
                $messages = @()
                if ($SystemPrompt) {
                    $messages += @{ role = "user"; content = $SystemPrompt }
                    $messages += @{ role = "assistant"; content = "Understood. I'll follow these instructions." }
                }
                $messages += @{ role = "user"; content = $Prompt }
                
                @{
                    model       = $Model
                    messages    = $messages
                    max_tokens  = $MaxTokens
                    temperature = $Temperature
                } | ConvertTo-Json -Depth 10
            }
            
            "Google" {
                $contents = @()
                if ($SystemPrompt) {
                    $contents += @{ role = "user"; parts = @{ text = $SystemPrompt } }
                    $contents += @{ role = "model"; parts = @{ text = "Understood. I'll follow these instructions." } }
                }
                $contents += @{ role = "user"; parts = @{ text = $Prompt } }
                
                @{
                    contents         = $contents
                    generationConfig = @{
                        maxOutputTokens = $MaxTokens
                        temperature     = $Temperature
                    }
                } | ConvertTo-Json -Depth 10
            }
            
            "Azure" {
                $messages = @()
                if ($SystemPrompt) {
                    $messages += @{ role = "system"; content = $SystemPrompt }
                }
                $messages += @{ role = "user"; content = $Prompt }
                
                @{
                    messages    = $messages
                    max_tokens  = $MaxTokens
                    temperature = $Temperature
                } | ConvertTo-Json -Depth 10
            }
            
            "Cohere" {
                @{
                    model       = $Model
                    prompt      = $Prompt
                    max_tokens  = $MaxTokens
                    temperature = $Temperature
                } | ConvertTo-Json -Depth 10
            }
            
            "HuggingFace" {
                @{
                    inputs     = $Prompt
                    parameters = @{
                        max_new_tokens = $MaxTokens
                        temperature    = $Temperature
                    }
                } | ConvertTo-Json -Depth 10
            }
        }
        
        # Set headers based on provider
        $headers = switch ($Provider) {
            "OpenAI" {
                @{
                    "Content-Type"  = "application/json"
                    "Authorization" = "Bearer $ApiKey"
                }
            }
            
            "Anthropic" {
                @{
                    "Content-Type"      = "application/json"
                    "x-api-key"         = $ApiKey
                    "anthropic-version" = "2023-06-01"
                }
            }
            
            "Google" {
                @{
                    "Content-Type" = "application/json"
                }
            }
            
            "Azure" {
                @{
                    "Content-Type" = "application/json"
                    "api-key"      = $ApiKey
                }
            }
            
            "Cohere" {
                @{
                    "Content-Type"  = "application/json"
                    "Authorization" = "Bearer $ApiKey"
                }
            }
            
            "HuggingFace" {
                @{
                    "Content-Type"  = "application/json"
                    "Authorization" = "Bearer $ApiKey"
                }
            }
        }
        
        # Add API key to URL for Google
        if ($Provider -eq "Google") {
            $Endpoint += "?key=$ApiKey"
        }
        
        # Make the API call
        Write-Verbose "Calling $Provider API at $Endpoint"
        $response = Invoke-RestMethod -Uri $Endpoint -Method Post -Headers $headers -Body $body -TimeoutSec 30
        
        # Extract response content based on provider
        $content = switch ($Provider) {
            "OpenAI" { $response.choices[0].message.content }
            "Anthropic" { $response.content[0].text }
            "Google" { $response.candidates[0].content.parts[0].text }
            "Azure" { $response.choices[0].message.content }
            "Cohere" { $response.generations[0].text }
            "HuggingFace" { $response[0].generated_text }
        }
        
        # Return custom object with response
        $result = [PSCustomObject]@{
            Provider   = $Provider
            Model      = $Model
            Prompt     = $Prompt
            Response   = $content.Trim()
            TokensUsed = if ($response.usage) { $response.usage.total_tokens } else { $null }
            Timestamp  = Get-Date
        }
        
        if ($ResponseOnly) {
            return $content.Trim()
        }
        elseif ($FullResponse) {
            return $result | Format-List -Property *
        }
        else {
            # Default to Format-List for better readability
            return $result | Format-List -Property *
        }
        
    }
    catch {
        $errorMessage = "Error calling $Provider API: $($_.Exception.Message)"
        Write-Error $errorMessage
        
        # Return error object
        return [PSCustomObject]@{
            Provider  = $Provider
            Model     = $Model
            Prompt    = $Prompt
            Response  = $null
            Error     = $errorMessage
            Timestamp = Get-Date
        }
    }
}
