/**
 * LocalLLM (Ollama) translation service provider for Rocket.Chat AutoTranslate.
 *
 * This file replaces the former Google Translate provider (googleTranslate.ts).
 * The overall structure — class shape, registration pattern, deTokenize calls,
 * and the per-language loop — is intentionally retained from the Google implementation
 * to minimise diff noise. Only the HTTP call and response parsing have changed.
 *
 * Key differences from Google Translate:
 *  - No external API key required; the Ollama instance runs locally.
 *  - Supported languages are hardcoded (no /languages endpoint to fetch from).
 *  - One POST to /api/generate per target language, with stream:false so we receive
 *    a single JSON object instead of a stream of newline-delimited chunks.
 *  - Response is in body.response (plain string) instead of body.data.translations[].
 */

import type { IMessage, IProviderMetadata, ISupportedLanguage, ITranslationResult, MessageAttachment } from '@rocket.chat/core-typings';
import { serverFetch as fetch } from '@rocket.chat/server-fetch';

import { AutoTranslate, TranslationProviderRegistry } from './autotranslate';
import { SystemLogger } from '../../../server/lib/logger/system';
import { settings } from '../../settings/server';

// ---------------------------------------------------------------------------
// Language definitions
// ---------------------------------------------------------------------------

/**
 * Full language names used inside the translation prompt.
 * LLMs respond more reliably to language names than to ISO codes.
 * Extend this map if additional languages are added to SUPPORTED_LANGUAGES.
 */
const LANGUAGE_NAMES: Record<string, string> = {
	en: 'English',
	zh: 'Chinese',
	vi: 'Vietnamese',
	id: 'Indonesian',
};

/**
 * The fixed list of languages this provider exposes to Rocket.Chat.
 * Replaces the dynamic fetch that Google/DeepL/Microsoft perform against their
 * respective /languages endpoints — Ollama has no such endpoint.
 */
const SUPPORTED_LANGUAGES: ISupportedLanguage[] = [
	{ language: 'en', name: 'English' },
	{ language: 'zh', name: 'Chinese' },
	{ language: 'vi', name: 'Vietnamese' },
	{ language: 'id', name: 'Indonesian' },
];

// ---------------------------------------------------------------------------
// Provider class
// ---------------------------------------------------------------------------

/**
 * LocalLLM translation provider — routes translation requests to a locally-running
 * Ollama instance instead of a cloud API.
 *
 * Admin settings consumed:
 *  - AutoTranslate_LocalLLMBaseURL  (e.g. http://localhost:11434)
 *  - AutoTranslate_LocalLLMModel    (e.g. qwen3.5:9b)
 *
 * @class
 * @augments AutoTranslate
 */
class LocalLLMAutoTranslate extends AutoTranslate {
	/** Base URL of the running Ollama instance, without trailing slash. */
	baseUrl: string;

	/** Ollama model name to use for all translation requests. */
	model: string;

	/**
	 * Registers admin-setting watchers so that URL and model changes in the
	 * Rocket.Chat admin panel take effect immediately without a server restart.
	 * @constructor
	 */
	constructor() {
		super();

		// Must match the `key` value in the AutoTranslate_ServiceProvider dropdown
		// defined in server/settings/message.ts
		this.name = 'localllm-translate';

		// Watch the admin setting for the Ollama base URL (e.g. http://localhost:11434)
		settings.watch<string>('AutoTranslate_LocalLLMBaseURL', (value) => {
			this.baseUrl = value;
		});

		// Watch the admin setting for the model name (e.g. qwen3.5:9b)
		settings.watch<string>('AutoTranslate_LocalLLMModel', (value) => {
			this.model = value;
		});
	}

	// -------------------------------------------------------------------------
	// Interface implementation — metadata & settings
	// -------------------------------------------------------------------------

	/**
	 * Returns metadata shown in the AutoTranslate settings UI.
	 * @private implements super abstract method.
	 */
	_getProviderMetadata(): IProviderMetadata {
		return {
			name: this.name,
			// Display name shown in the admin dropdown
			displayName: 'LocalLLM',
			settings: this._getSettings(),
		};
	}

	/**
	 * Exposes current runtime settings so they can be read back by the UI.
	 * We reuse the apiKey/apiEndPointUrl field names from the IProviderMetadata
	 * interface — here apiKey holds the baseUrl and apiEndPointUrl the full path.
	 * @private implements super abstract method.
	 */
	_getSettings(): IProviderMetadata['settings'] {
		return {
			apiKey: this.baseUrl,
			apiEndPointUrl: `${this.baseUrl}/api/generate`,
		};
	}

	// -------------------------------------------------------------------------
	// Interface implementation — language support
	// -------------------------------------------------------------------------

	/**
	 * Returns the hardcoded list of supported languages.
	 *
	 * Unlike Google/Microsoft/DeepL, Ollama has no /languages endpoint, so we
	 * return a fixed list. The `target` parameter is kept for interface compatibility
	 * but is not used — all callers receive the same list regardless.
	 *
	 * @private implements super abstract method.
	 * @param {string} _target — unused; kept for interface compatibility
	 */
	async getSupportedLanguages(_target: string): Promise<ISupportedLanguage[]> {
		// No API key guard needed — Ollama requires no authentication
		return SUPPORTED_LANGUAGES;
	}

	// -------------------------------------------------------------------------
	// Core translation logic
	// -------------------------------------------------------------------------

	/**
	 * Fires a single POST to Ollama's /api/generate endpoint and returns the
	 * translated text string.
	 *
	 * stream is set to false so the response arrives as one complete JSON object:
	 *   { "model": "...", "response": "<translated text>", ... }
	 * If stream were true (Ollama default), the response body would be a series of
	 * newline-delimited JSON chunks that would require manual stream parsing.
	 *
	 * @private
	 * @param {string} text     — plain text to translate (already tokenized by base class)
	 * @param {string} language — ISO 639-1 target language code
	 * @throws {Error} if the HTTP response is not OK
	 * @returns {Promise<string>} translated text from body.response
	 */
	private async _callOllama(text: string, language: string): Promise<string> {
		// Resolve human-readable language name for the prompt; fall back to the code
		// if the language is somehow not in the map (e.g. an unlisted code)
		const languageName = LANGUAGE_NAMES[language] ?? language;

		// Instruct the model to return only the translated output.
		// The explicit instruction prevents the model from adding explanations,
		// notes, or alternative translations that would pollute the stored result.
		const prompt = `Translate the following text to ${languageName}. Return only the translated text, no explanations or extra formatting:\n\n${text}`;

		// SECURITY: ignoreSsrfValidation is required because the target is a
		// local/private-network address (e.g. http://localhost:11434).
		// The URL originates from an admin-only setting, not from user input.
		const result = await fetch(`${this.baseUrl}/api/generate`, {
			ignoreSsrfValidation: true,
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				model: this.model,
				prompt,
				// Disable streaming: receive one complete JSON response instead of chunks
				stream: false,
			}),
			// Default server-fetch timeout is 20s which is too short for a local LLM.
			// 120s gives the model enough time to translate even longer messages.
			timeout: 120000,
		});

		if (!result.ok) {
			throw new Error(result.statusText);
		}

		const body = await result.json();

		// Ollama /api/generate response shape (stream:false):
		// { "model": string, "response": string, "done": true, ... }
		return (body.response as string) ?? '';
	}

	/**
	 * Translates a message into each requested target language.
	 *
	 * Iterates over targetLanguages and fires one Ollama request per language,
	 * matching the per-language loop pattern used by the former Google provider.
	 * Multi-line messages are sent as a single prompt — Ollama handles newlines
	 * natively, so the line-splitting that Google required (into separate `q` params)
	 * is no longer necessary.
	 *
	 * Language codes containing a region subtag (e.g. zh-CN) are normalised to their
	 * two-character base code (zh) when the full code is not in the supported list.
	 *
	 * @private implements super abstract method.
	 * @param {IMessage}  message
	 * @param {string[]}  targetLanguages
	 * @returns {ITranslationResult} map of language code → translated string
	 */
	async _translateMessage(message: IMessage, targetLanguages: string[]): Promise<ITranslationResult> {
		const translations: { [k: string]: string } = {};

		for (let language of targetLanguages) {
			// Normalise region-subtag codes (e.g. zh-CN → zh) when the full code is
			// not in the supported language list, mirroring the Google/DeepL behaviour
			if (language.indexOf('-') !== -1 && !SUPPORTED_LANGUAGES.find((l) => l.language === language)) {
				language = language.slice(0, 2);
			}

			try {
				const translatedText = await this._callOllama(message.msg, language);

				// deTokenize restores the placeholders (URLs, @mentions, :emoji:, code blocks)
				// that the base class substituted before handing the text to us for translation
				translations[language] = this.deTokenize(Object.assign({}, message, { msg: translatedText }));
			} catch (err) {
				SystemLogger.error({ msg: 'LocalLLM: Error translating message', err });
			}
		}

		return translations;
	}

	/**
	 * Translates message attachment descriptions into each target language.
	 * Uses the same _callOllama helper as _translateMessage.
	 * Note: deTokenize is not called here because attachment descriptions go through
	 * a different path in the base class that does not tokenize them beforehand.
	 *
	 * @private implements super abstract method.
	 * @param {MessageAttachment} attachment
	 * @param {string[]}          targetLanguages
	 * @returns {ITranslationResult} map of language code → translated string
	 */
	async _translateAttachmentDescriptions(attachment: MessageAttachment, targetLanguages: string[]): Promise<ITranslationResult> {
		const translations: { [k: string]: string } = {};

		// Prefer description over text, fall back to empty string if neither is set
		const text = attachment.description || attachment.text || '';

		for (let language of targetLanguages) {
			// Same region-subtag normalisation as in _translateMessage
			if (language.indexOf('-') !== -1 && !SUPPORTED_LANGUAGES.find((l) => l.language === language)) {
				language = language.slice(0, 2);
			}

			try {
				translations[language] = await this._callOllama(text, language);
			} catch (err) {
				SystemLogger.error({ msg: 'LocalLLM: Error translating attachment description', err });
			}
		}

		return translations;
	}
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

// Register LocalLLM as a translation provider so TranslationProviderRegistry
// can activate it when AutoTranslate_ServiceProvider is set to 'localllm-translate'.
// This mirrors the registration pattern used by googleTranslate.ts, deeplTranslate.ts,
// and msTranslate.ts.
TranslationProviderRegistry.registerProvider(new LocalLLMAutoTranslate());
