//+------------------------------------------------------------------+
//| ExportBrokerSpecs.mq5                                            |
//| One-off export of everything the Python cost/execution model     |
//| needs from a broker. Run once per account (FBS and Clear).       |
//|                                                                  |
//| Writes tab-separated files to <Data Folder>\MQL5\Files\ :        |
//|   <Company>_account.tsv     account + server-time offset         |
//|   <Company>_symbols.tsv     contract/tick/swap/margin specs      |
//|   <Company>_sessions.tsv    trading sessions per weekday         |
//|   <Company>_commissions.tsv commission/swap/fee per symbol,      |
//|                             aggregated from your deal history    |
//|   <Company>_calendar.tsv    economic calendar (server time)      |
//+------------------------------------------------------------------+
#property copyright "DataScienceAndTrading"
#property version   "1.00"
#property script_show_inputs

input string   InpSymbols        = "";              // Comma-separated symbols; empty = all
input bool     InpOnlyVisible    = true;            // With empty list: Market Watch only
input bool     InpExportCalendar = true;            // Export economic calendar history
input datetime InpCalendarFrom   = D'2016.01.01';   // Calendar start
input int      InpCommissionDays = 365;             // Deal-history look-back for commissions

//--- helpers --------------------------------------------------------
string Sanitize(const string s)
  {
   string out = "";
   for(int i = 0; i < StringLen(s); i++)
     {
      ushort ch = StringGetCharacter(s, i);
      bool alnum = (ch >= '0' && ch <= '9') || (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');
      if(alnum)
         out += ShortToString(ch);
      else
         if(StringLen(out) > 0 && StringSubstr(out, StringLen(out) - 1) != "_")
            out += "_";
     }
   return out;
  }

string Ts(const datetime t) { return TimeToString(t, TIME_DATE | TIME_SECONDS); }

// Calendar numeric fields are long * 1e6; LONG_MIN means "not set".
string CalVal(const long v) { return v == LONG_MIN ? "" : DoubleToString(v / 1000000.0, 6); }

int OpenOut(const string name)
  {
   int fh = FileOpen(name, FILE_WRITE | FILE_CSV | FILE_ANSI, '\t');
   if(fh == INVALID_HANDLE)
      PrintFormat("Cannot open %s (error %d)", name, GetLastError());
   return fh;
  }

int CollectSymbols(string &syms[])
  {
   ArrayResize(syms, 0);
   if(StringLen(InpSymbols) > 0)
     {
      string parts[];
      int n = StringSplit(InpSymbols, ',', parts);
      for(int i = 0; i < n; i++)
        {
         string s = parts[i];
         StringTrimLeft(s);
         StringTrimRight(s);
         if(s == "")
            continue;
         int k = ArraySize(syms);
         ArrayResize(syms, k + 1);
         syms[k] = s;
        }
      return ArraySize(syms);
     }
   int total = SymbolsTotal(InpOnlyVisible);
   ArrayResize(syms, total);
   for(int i = 0; i < total; i++)
      syms[i] = SymbolName(i, InpOnlyVisible);
   return total;
  }

//--- exports ----------------------------------------------------------
void ExportAccount(const string prefix)
  {
   int fh = OpenOut(prefix + "_account.tsv");
   if(fh == INVALID_HANDLE)
      return;
   datetime server = TimeTradeServer();
   datetime gmt    = TimeGMT();
   // Rounded to 30 min: TimeTradeServer is an estimate, not exact to the second.
   double offset_h = MathRound((double)(server - gmt) / 1800.0) / 2.0;
   FileWrite(fh, "key", "value");
   FileWrite(fh, "company", AccountInfoString(ACCOUNT_COMPANY));
   FileWrite(fh, "server", AccountInfoString(ACCOUNT_SERVER));
   FileWrite(fh, "currency", AccountInfoString(ACCOUNT_CURRENCY));
   FileWrite(fh, "leverage", AccountInfoInteger(ACCOUNT_LEVERAGE));
   FileWrite(fh, "trade_mode", EnumToString((ENUM_ACCOUNT_TRADE_MODE)AccountInfoInteger(ACCOUNT_TRADE_MODE)));
   FileWrite(fh, "margin_mode", EnumToString((ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE)));
   FileWrite(fh, "stopout_mode", EnumToString((ENUM_ACCOUNT_STOPOUT_MODE)AccountInfoInteger(ACCOUNT_MARGIN_SO_MODE)));
   FileWrite(fh, "margin_call_level", AccountInfoDouble(ACCOUNT_MARGIN_SO_CALL));
   FileWrite(fh, "stopout_level", AccountInfoDouble(ACCOUNT_MARGIN_SO_SO));
   FileWrite(fh, "server_time_now", Ts(server));
   FileWrite(fh, "gmt_time_now", Ts(gmt));
   FileWrite(fh, "server_gmt_offset_hours", DoubleToString(offset_h, 1));
   FileWrite(fh, "local_pc_dst_active", TimeDaylightSavings() != 0 ? "true" : "false");
   FileWrite(fh, "terminal_build", TerminalInfoInteger(TERMINAL_BUILD));
   FileClose(fh);
  }

void ExportSymbols(const string prefix, const string &syms[])
  {
   int fh = OpenOut(prefix + "_symbols.tsv");
   int fs = OpenOut(prefix + "_sessions.tsv");
   if(fh == INVALID_HANDLE || fs == INVALID_HANDLE)
      return;
   FileWrite(fh, "symbol", "path", "description", "calc_mode", "trade_mode", "digits", "point",
             "tick_size", "tick_value", "tick_value_profit", "tick_value_loss", "contract_size",
             "currency_base", "currency_profit", "currency_margin",
             "volume_min", "volume_max", "volume_step",
             "spread_now_points", "spread_float", "stops_level", "freeze_level",
             "swap_mode", "swap_long", "swap_short", "swap_3day",
             "margin_initial", "margin_maintenance", "margin_hedged",
             "margin_rate_init_buy", "margin_rate_maint_buy", "margin_1lot_buy_now", "filling_mode");
   FileWrite(fs, "symbol", "weekday", "session_idx", "trade_from_sec", "trade_to_sec");

   for(int i = 0; i < ArraySize(syms); i++)
     {
      string s = syms[i];
      if(!SymbolInfoInteger(s, SYMBOL_EXIST))
        {
         PrintFormat("Skipping unknown symbol '%s'", s);
         continue;
        }
      double mr_init = 0, mr_maint = 0, margin_1lot = 0;
      SymbolInfoMarginRate(s, ORDER_TYPE_BUY, mr_init, mr_maint);
      double ask = SymbolInfoDouble(s, SYMBOL_ASK);
      string margin_str = "";
      if(ask > 0 && OrderCalcMargin(ORDER_TYPE_BUY, s, 1.0, ask, margin_1lot))
         margin_str = DoubleToString(margin_1lot, 2);

      FileWrite(fh, s,
                SymbolInfoString(s, SYMBOL_PATH),
                SymbolInfoString(s, SYMBOL_DESCRIPTION),
                EnumToString((ENUM_SYMBOL_CALC_MODE)SymbolInfoInteger(s, SYMBOL_TRADE_CALC_MODE)),
                EnumToString((ENUM_SYMBOL_TRADE_MODE)SymbolInfoInteger(s, SYMBOL_TRADE_MODE)),
                SymbolInfoInteger(s, SYMBOL_DIGITS),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_POINT), 10),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_TRADE_TICK_SIZE), 10),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_TRADE_TICK_VALUE), 10),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_TRADE_TICK_VALUE_PROFIT), 10),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_TRADE_TICK_VALUE_LOSS), 10),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_TRADE_CONTRACT_SIZE), 4),
                SymbolInfoString(s, SYMBOL_CURRENCY_BASE),
                SymbolInfoString(s, SYMBOL_CURRENCY_PROFIT),
                SymbolInfoString(s, SYMBOL_CURRENCY_MARGIN),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_VOLUME_MIN), 4),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_VOLUME_MAX), 4),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_VOLUME_STEP), 4),
                SymbolInfoInteger(s, SYMBOL_SPREAD),
                SymbolInfoInteger(s, SYMBOL_SPREAD_FLOAT) ? "true" : "false",
                SymbolInfoInteger(s, SYMBOL_TRADE_STOPS_LEVEL),
                SymbolInfoInteger(s, SYMBOL_TRADE_FREEZE_LEVEL),
                EnumToString((ENUM_SYMBOL_SWAP_MODE)SymbolInfoInteger(s, SYMBOL_SWAP_MODE)),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_SWAP_LONG), 6),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_SWAP_SHORT), 6),
                EnumToString((ENUM_DAY_OF_WEEK)SymbolInfoInteger(s, SYMBOL_SWAP_ROLLOVER3DAYS)),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_MARGIN_INITIAL), 4),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_MARGIN_MAINTENANCE), 4),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_MARGIN_HEDGED), 4),
                DoubleToString(mr_init, 6),
                DoubleToString(mr_maint, 6),
                margin_str,
                SymbolInfoInteger(s, SYMBOL_FILLING_MODE));

      for(int d = 0; d < 7; d++)
        {
         datetime from = 0, to = 0;
         for(uint idx = 0; SymbolInfoSessionTrade(s, (ENUM_DAY_OF_WEEK)d, idx, from, to); idx++)
            FileWrite(fs, s, EnumToString((ENUM_DAY_OF_WEEK)d), idx, (long)from, (long)to);
        }
     }
   FileClose(fh);
   FileClose(fs);
  }

void ExportCommissions(const string prefix)
  {
   int fh = OpenOut(prefix + "_commissions.tsv");
   if(fh == INVALID_HANDLE)
      return;
   FileWrite(fh, "symbol", "deals", "lots_opened", "commission_total", "commission_per_lot_roundtrip",
             "swap_total", "fee_total", "first_deal", "last_deal");
   datetime now = TimeCurrent();
   if(!HistorySelect(now - (datetime)InpCommissionDays * 86400, now))
     {
      Print("HistorySelect failed: ", GetLastError());
      FileClose(fh);
      return;
     }
   string   sym[];
   int      cnt[];
   double   lots[], comm[], swp[], fee[];
   datetime first[], last[];
   int total = HistoryDealsTotal();
   for(int i = 0; i < total; i++)
     {
      ulong tk = HistoryDealGetTicket(i);
      string s = HistoryDealGetString(tk, DEAL_SYMBOL);
      if(s == "")
         continue;   // balance/credit operations
      int k = -1;
      for(int j = 0; j < ArraySize(sym); j++)
         if(sym[j] == s)
           {
            k = j;
            break;
           }
      if(k < 0)
        {
         k = ArraySize(sym);
         ArrayResize(sym, k + 1); ArrayResize(cnt, k + 1); ArrayResize(lots, k + 1);
         ArrayResize(comm, k + 1); ArrayResize(swp, k + 1); ArrayResize(fee, k + 1);
         ArrayResize(first, k + 1); ArrayResize(last, k + 1);
         sym[k] = s; cnt[k] = 0; lots[k] = 0; comm[k] = 0; swp[k] = 0; fee[k] = 0;
         first[k] = (datetime)HistoryDealGetInteger(tk, DEAL_TIME);
        }
      cnt[k]++;
      if((ENUM_DEAL_ENTRY)HistoryDealGetInteger(tk, DEAL_ENTRY) == DEAL_ENTRY_IN)
         lots[k] += HistoryDealGetDouble(tk, DEAL_VOLUME);
      comm[k] += HistoryDealGetDouble(tk, DEAL_COMMISSION);
      swp[k]  += HistoryDealGetDouble(tk, DEAL_SWAP);
      fee[k]  += HistoryDealGetDouble(tk, DEAL_FEE);
      last[k]  = (datetime)HistoryDealGetInteger(tk, DEAL_TIME);
     }
   for(int k = 0; k < ArraySize(sym); k++)
      FileWrite(fh, sym[k], cnt[k], DoubleToString(lots[k], 2), DoubleToString(comm[k], 2),
                lots[k] > 0 ? DoubleToString(comm[k] / lots[k], 4) : "",
                DoubleToString(swp[k], 2), DoubleToString(fee[k], 2), Ts(first[k]), Ts(last[k]));
   FileClose(fh);
  }

void ExportCalendar(const string prefix)
  {
   int fh = OpenOut(prefix + "_calendar.tsv");
   if(fh == INVALID_HANDLE)
      return;
   FileWrite(fh, "value_id", "time_server", "country", "currency", "event_id", "event_name",
             "importance", "time_mode", "event_type", "unit", "multiplier", "period", "revision",
             "actual", "forecast", "previous", "revised_previous", "impact");
   const int step = 90 * 86400;   // quarterly chunks keep the value array small
   datetime now = TimeTradeServer();
   long rows = 0;
   for(datetime from = InpCalendarFrom; from < now; from += step)
     {
      MqlCalendarValue vals[];
      int n = CalendarValueHistory(vals, from, from + step - 1);
      for(int i = 0; i < n; i++)
        {
         MqlCalendarEvent ev;
         MqlCalendarCountry c;
         if(!CalendarEventById(vals[i].event_id, ev) || !CalendarCountryById(ev.country_id, c))
            continue;
         FileWrite(fh, vals[i].id, Ts(vals[i].time), c.code, c.currency, ev.id, ev.name,
                   EnumToString(ev.importance), EnumToString(ev.time_mode), EnumToString(ev.type),
                   EnumToString(ev.unit), EnumToString(ev.multiplier),
                   TimeToString(vals[i].period, TIME_DATE), vals[i].revision,
                   CalVal(vals[i].actual_value), CalVal(vals[i].forecast_value),
                   CalVal(vals[i].prev_value), CalVal(vals[i].revised_prev_value),
                   EnumToString(vals[i].impact_type));
         rows++;
        }
     }
   FileClose(fh);
   PrintFormat("Calendar: %I64d rows exported", rows);
  }

//+------------------------------------------------------------------+
void OnStart()
  {
   string prefix = Sanitize(AccountInfoString(ACCOUNT_COMPANY));
   if(prefix == "")
      prefix = "broker";
   string syms[];
   CollectSymbols(syms);

   ExportAccount(prefix);
   ExportSymbols(prefix, syms);
   ExportCommissions(prefix);
   if(InpExportCalendar)
      ExportCalendar(prefix);

   PrintFormat("Done. %d symbols. Files in %s\\MQL5\\Files\\%s_*.tsv",
               ArraySize(syms), TerminalInfoString(TERMINAL_DATA_PATH), prefix);
  }
//+------------------------------------------------------------------+
