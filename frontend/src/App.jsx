import React, { useState, useEffect, useRef } from 'react';
import { 
  Languages, 
  Send, 
  Copy, 
  History, 
  Trash2, 
  Sparkles, 
  Check, 
  AlertCircle, 
  ArrowRightLeft,
  Clock,
  ExternalLink
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_URL = `${window.location.protocol}//${window.location.hostname}:5500/api/translate`;

export default function App() {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);
  const [history, setHistory] = useState([]);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState(null);
  const [showHistory, setShowHistory] = useState(false);
  const [stats, setStats] = useState(null);

  // Load history from localStorage
  useEffect(() => {
    const savedHistory = localStorage.getItem('translation_history');
    if (savedHistory) {
      setHistory(JSON.parse(savedHistory));
    }
  }, []);

  // Save history to localStorage
  useEffect(() => {
    localStorage.setItem('translation_history', JSON.stringify(history));
  }, [history]);

  const handleTranslate = async () => {
    if (!inputText.trim() || isTranslating) return;

    setIsTranslating(true);
    setError(null);
    setStats(null);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText }),
      });

      const data = await response.json();

      if (response.ok) {
        setOutputText(data.translation);
        setStats({ time: data.time_ms });
        
        // Add to history
        const newEntry = {
          id: Date.now(),
          src: inputText,
          tgt: data.translation,
          timestamp: new Date().toLocaleTimeString(),
          timeMs: data.time_ms
        };
        setHistory(prev => [newEntry, ...prev].slice(0, 20));
      } else {
        setError(data.error || 'Translation failed. Is the API server running?');
      }
    } catch (err) {
      setError('Could not connect to the API server. Please make sure api_server.py is running.');
    } finally {
      setIsTranslating(false);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(outputText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('translation_history');
  };

  const useFromHistory = (entry) => {
    setInputText(entry.src);
    setOutputText(entry.tgt);
    setShowHistory(false);
  };

  const handleKeyPress = (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
      handleTranslate();
    }
  };

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center p-4 md:p-8">
      {/* Background Decor */}
      <div className="bg-gradient" />
      <div className="bg-animate top-[-10%] left-[-10%]" />
      <div className="bg-animate bottom-[-10%] right-[-10%] !bg-secondary" style={{ animationDelay: '-10s' }} />

      {/* Main Container */}
      <motion.div 
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="w-full max-w-5xl glass-panel p-6 md:p-10 flex flex-col gap-8"
      >
        {/* Header */}
        <header className="flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-2xl bg-primary/20 border border-primary/30">
              <Languages className="w-8 h-8 text-primary" />
            </div>
            <div>
              <h1 className="text-3xl font-bold logo-text tracking-tight">TRANSLATOR.AI</h1>
              <p className="text-xs text-text-secondary uppercase tracking-[3px] mt-1">English ➔ Hindi Transformer</p>
            </div>
          </div>
          <button 
            onClick={() => setShowHistory(!showHistory)}
            className="p-3 rounded-xl border border-border-glass hover:bg-white/5 transition-all text-text-secondary flex items-center gap-2 group"
          >
            <History className="w-5 h-5 group-hover:rotate-[-45deg] transition-transform" />
            <span className="hidden sm:inline font-medium">History</span>
          </button>
        </header>

        {/* Translation Area */}
        <div className="grid grid-cols-1 md:grid-cols-[1fr,auto,1fr] items-center gap-6">
          {/* Input */}
          <div className="flex flex-col gap-3">
            <div className="flex justify-between items-center px-2">
              <span className="text-sm font-semibold text-text-secondary flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary" /> ENGLISH
              </span>
              <span className="text-xs text-text-secondary/50 font-mono">{inputText.length} / 500</span>
            </div>
            <textarea
              className="input-glow h-48 md:h-64"
              placeholder="Enter English text to translate..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyPress}
            />
          </div>

          {/* Center Action */}
          <div className="flex flex-col items-center gap-4">
            <div className="p-3 rounded-full border border-border-glass bg-white/5 hidden md:flex">
              <ArrowRightLeft className="w-6 h-6 text-text-secondary" />
            </div>
            <button 
              onClick={handleTranslate}
              disabled={isTranslating || !inputText.trim()}
              className="btn-premium flex items-center gap-3 w-full md:w-auto overflow-hidden group"
            >
              {isTranslating ? (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>WAIT...</span>
                </div>
              ) : (
                <>
                  <Sparkles className="w-5 h-5 group-hover:scale-125 transition-transform" />
                  <span>TRANSLATE</span>
                  <Send className="w-4 h-4 ml-1 opacity-70" />
                </>
              )}
            </button>
          </div>

          {/* Output */}
          <div className="flex flex-col gap-3">
            <div className="flex justify-between items-center px-2">
              <span className="text-sm font-semibold text-text-secondary flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-accent" /> HINDI (हिन्दी)
              </span>
              {stats && (
                <span className="text-[10px] text-accent/70 font-mono flex items-center gap-1">
                  <Clock className="w-3 h-3" /> {stats.time}ms
                </span>
              )}
            </div>
            <div className="relative group">
              <textarea
                className="input-glow h-48 md:h-64 !bg-black/20 !cursor-default"
                placeholder="Translation will appear here..."
                value={outputText}
                readOnly
              />
              <AnimatePresence>
                {outputText && (
                  <motion.button
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    onClick={copyToClipboard}
                    className="absolute bottom-4 right-4 p-3 rounded-xl bg-primary text-white shadow-lg hover:scale-110 active:scale-95 transition-all"
                  >
                    {copied ? <Check className="w-5 h-5" /> : <Copy className="w-5 h-5" />}
                  </motion.button>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>

        {/* Footer Info */}
        <footer className="flex flex-col sm:flex-row justify-between items-center pt-4 border-t border-border-glass gap-4">
          <div className="flex items-center gap-4 text-xs text-text-secondary/60">
            <span className="flex items-center gap-1"><div className="w-1.5 h-1.5 rounded-full bg-green-500" /> System Active</span>
            <span className="flex items-center gap-1"><div className="w-1.5 h-1.5 rounded-full bg-primary" /> Transformer V1.0</span>
          </div>
          <p className="text-xs text-text-secondary/60 italic font-mono uppercase tracking-widest">
            Ctrl + Enter to fast-translate
          </p>
        </footer>
      </motion.div>

      {/* Error Toast */}
      <AnimatePresence>
        {error && (
          <motion.div 
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="fixed bottom-10 px-6 py-4 rounded-2xl bg-red-500/10 border border-red-500/20 backdrop-blur-xl flex items-center gap-4 text-red-200 shadow-2xl z-50"
          >
            <AlertCircle className="w-6 h-6 text-red-500" />
            <span className="font-medium">{error}</span>
            <button onClick={() => setError(null)} className="ml-2 opacity-50 hover:opacity-100 font-bold">✕</button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* History Sidebar */}
      <AnimatePresence>
        {showHistory && (
          <>
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowHistory(false)}
              className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
            />
            <motion.div 
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: "spring", damping: 25, stiffness: 200 }}
              className="fixed right-0 top-0 bottom-0 w-full max-w-sm bg-bg-deep border-l border-border-glass p-8 z-[60] shadow-2xl overflow-y-auto"
            >
              <div className="flex justify-between items-center mb-10">
                <h2 className="text-2xl font-bold flex items-center gap-3">
                  <History className="text-primary" /> History
                </h2>
                <button 
                  onClick={() => setShowHistory(false)}
                  className="p-2 hover:bg-white/5 rounded-full text-text-secondary transition-colors"
                >
                  ✕
                </button>
              </div>

              {history.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-20 text-text-secondary gap-4">
                  <Clock className="w-12 h-12 opacity-20" />
                  <p className="font-medium opacity-50">No translations yet</p>
                </div>
              ) : (
                <div className="flex flex-col gap-4">
                  {history.map((item) => (
                    <motion.div 
                      key={item.id}
                      whileHover={{ scale: 1.02 }}
                      className="p-4 rounded-2xl border border-border-glass bg-white/5 hover:border-primary/30 cursor-pointer transition-all group"
                      onClick={() => useFromHistory(item)}
                    >
                      <div className="text-[10px] text-text-secondary flex justify-between mb-2">
                        <span className="uppercase tracking-widest">{item.timestamp}</span>
                        <span className="font-mono text-primary">{item.timeMs}ms</span>
                      </div>
                      <p className="text-sm font-medium mb-1 line-clamp-1">{item.src}</p>
                      <p className="text-sm text-text-secondary group-hover:text-accent transition-colors line-clamp-1">{item.tgt}</p>
                    </motion.div>
                  ))}
                  
                  <button 
                    onClick={clearHistory}
                    className="mt-6 w-full p-4 rounded-2xl border border-red-500/20 text-red-500 font-bold flex items-center justify-center gap-2 hover:bg-red-500/10 transition-all uppercase tracking-widest text-xs"
                  >
                    <Trash2 className="w-4 h-4" /> Clear All History
                  </button>
                </div>
              )}
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
